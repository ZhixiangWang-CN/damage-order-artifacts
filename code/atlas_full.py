"""Runner for PROTOCOL_ADDENDUM_8 (final batch).
  python3 atlas_full.py smoke  <model_key> <items_dir> <out_root>
  python3 atlas_full.py worker <items_dir> <out_root> <comma-separated model keys> <max_gb> <worker_name>
A worker claims (model, damage, seed) jobs for the listed models (claims are directories under <out_root>/claims, so all workers
of one machine share one queue), keeps the loaded model when it can, and never retries a failed job by itself.
env: ATLAS_TOK_BUDGET (tokens per forward batch, default 16000), ATLAS_QBITS (bits for qmix, default 3)
"""
import sys, os, json, gzip, random, re, time, math, glob, hashlib, traceback, unicodedata, gc, shutil
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from atlas import Damager
from atlas_fmt import INSTR as FMT_INSTR, FRE
TWICE_INSTR_N = 'Write the {n} that answers the question twice, separated by a comma and a space, and nothing else.'
TWICE_INSTR = 'Answer yes or no, written twice, separated by a comma and a space, and nothing else.'

MODELS = {'q15': ('Qwen/Qwen2.5-1.5B-Instruct', 5), 'llama': ('unsloth/Llama-3.2-3B-Instruct', 9), 'phi': ('microsoft/Phi-3.5-mini-instruct', 10),
          'q7': ('Qwen/Qwen2.5-7B-Instruct', 19), 'mis': ('mistralai/Mistral-7B-Instruct-v0.3', 18), 'gemma': ('unsloth/gemma-2-9b-it', 25), 'q14': ('Qwen/Qwen2.5-14B-Instruct', 34)}
NOCACHE = set()   # amended after smoke: Gemma uses cached generation once dropped layers are re-indexed (see Dmg.relayer)
SHORT = ['ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'COPY', 'SCIQ', 'BOOLQ']
OPTION = ['ARC', 'CSQA', 'MMLU_CALC', 'MMLU_MEM', 'HELLASWAG', 'PIQA', 'WINOGRANDE', 'OBQA']
TASKS = SHORT + OPTION
FORMATS = ['CHOICE', 'BARE', 'JSON', 'TWICE']
GEN_MX = {'BARE': 32, 'JSON': 32, 'TWICE': 32}   # BARE raised from 16 after smoke (answers were cut off)
UNT = ['noise', 'zero', 'neuron', 'ld_rand']; DAMAGES = UNT + ['ld_late', 'qmix']; STOCH = ('noise', 'zero', 'neuron', 'ld_rand', 'qmix')
GRID = {'noise': [0.03, 0.05, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6], 'zero': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'neuron': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], 'qmix': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],   # finer low end after smoke (2-bit: p = 0.5 already breaks every model)
        'ld_rand': [1, 2, 3, 4, 6, 8, 10, 12, 16], 'ld_late': [1, 2, 3, 4, 6, 8, 10, 12, 16]}
TARGETS = [0.95, 0.85, 0.75, 0.60, 0.45, 0.30, 0.18, 0.08]; LAYER_DOSES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
CALIB_SEEDS = (11, 12); MAIN_SEEDS = (1, 2, 3); KL_MAX = 1e-3; USABLE = 0.4
BUDGET = int(os.environ.get('ATLAS_TOK_BUDGET', 16000)); QBITS = int(os.environ.get('ATLAS_QBITS', 2))   # 2 bits by the protocol rule: Qwen-7B kept 0.73 > 0.3 at p = 1 with 3 bits in the smoke test

# ---------------------------------------------------------------- items, prompts, judging
def load_items(d):
    its = {t: json.load(open(f'{d}/{t}.json')) for t in TASKS + ['GSM8K_COT', 'PROBE']}
    return its, json.load(open(f'{d}/MANIFEST.json'))
def noun(it): return {'number': 'number', 'name': 'name', 'word': 'word', 'text': 'answer', 'letter': 'letter', 'yesno': 'answer (yes or no)'}[it['kind']]
def body(it, fmt):
    q = it['q'] + ('\n' + '\n'.join(f"{'ABCDE'[i]}. {o}" for i, o in enumerate(it['opts'])) if it['kind'] == 'letter' else '')
    if fmt in ('BARE', 'CHOICE'): ins = 'Answer with just yes or no.' if it['kind'] == 'yesno' else FMT_INSTR['F0'].format(n=noun(it))
    elif fmt == 'JSON': ins = FMT_INSTR['F2'].format(n=noun(it))
    else: ins = TWICE_INSTR if it['kind'] == 'yesno' else TWICE_INSTR_N.format(n=noun(it))   # amended after smoke: "the number" was read as an operand
    return f'{q}\n{ins}'
def target(it, fmt): a = it['ans']; return a if fmt == 'BARE' else ('{"answer": "%s"}' % a if fmt == 'JSON' else f'{a}, {a}')
def chance(it, fmt):
    k = len(it['cands'])
    return 1.0 / k if (fmt == 'CHOICE' or it['kind'] in ('letter', 'yesno')) else 0.0
def norm(s): return ' '.join(re.sub(r'[^a-z0-9]', ' ', ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower()).split())
def content(kind, v, gold, anywhere=False):
    if kind == 'number':
        if anywhere: return gold in [x.replace(',', '') for x in re.findall(r'-?\d[\d,]*', v)]
        m = re.search(r'-?\d[\d,]*', v); return bool(m) and m.group(0).replace(',', '') == gold
    if kind == 'yesno': m = re.search(r'\b(yes|no)\b', v.lower()); return bool(m) and m.group(1) == gold
    if kind == 'letter':
        m = re.search(r'(?<![A-Za-z])([A-E])(?![A-Za-z])', v) if anywhere else re.match(r'^\s*[\(\[]?([A-E])(?![A-Za-z])', v)
        return bool(m) and m.group(1) == gold
    g = norm(gold); return bool(g) and re.search(r'(?<![a-z0-9])' + re.escape(g) + r'(?![a-z0-9])', norm(v)) is not None
def judge(it, fmt, out):
    """returns (correct = format ok and content ok, format ok, content present anywhere in the output)."""
    s = out.strip()
    if fmt == 'BARE':   # amended after smoke: judge the first non-empty line (models often add an explanation after a blank line)
        val = next((l.strip() for l in s.split('\n') if l.strip()), ''); fok = bool(val)
    else:
        if fmt == 'TWICE': s = re.sub(r'\s*,\s*', ', ', s)   # amended after smoke: tolerate spaces around the comma
        m = FRE['F2' if fmt == 'JSON' else 'F5'].match(s); fok = bool(m); val = (m.group(1).strip('"') if m else s)
    return int(fok and content(it['kind'], val, it['ans'])), int(fok), int(content(it['kind'], s, it['ans'], anywhere=True))
def judge_cot(out, gold):
    m = re.findall(r'answer is[^\d\-]{0,12}(-?\d[\d,]*(?:\.\d+)?)', out.lower()) or re.findall(r'-?\d[\d,]*(?:\.\d+)?', out)
    if not m: return 0
    try: return int(abs(float(m[-1].replace(',', '').rstrip('.')) - float(gold)) < 1e-6)
    except ValueError: return 0

# ---------------------------------------------------------------- model, damage
def load_model(key):
    mid = MODELS[key][0]; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {'attn_implementation': 'eager'} if key == 'gemma' else {}
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, **kw).to('cuda').eval()
    if key in NOCACHE: model.generation_config.use_cache = False   # Gemma-2: cached generation crashes for some sets of removed layers
    txt = tok.apply_chat_template([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'ZQX'}], tokenize=False)
    ids = tok.encode(txt.split('ZQX')[-1], add_special_tokens=False); end = ids[0] if ids else tok.eos_token_id
    return tok, model, end
class Dmg(Damager):
    def __init__(s, model):
        super().__init__(model); s.cfg = model.config; s.orig_n = s.cfg.num_hidden_layers
        s.orig_types = list(getattr(s.cfg, 'layer_types', None) or []) or None; s.orig_idx = [l.self_attn.layer_idx for l in s.all_layers]
    def relayer(s):
        """after dropping layers: renumber the kept layers and shrink the config, so the KV cache (built from the config) matches them."""
        kept = list(s.model.model.layers)
        for i, l in enumerate(kept): l.self_attn.layer_idx = i
        s.cfg.num_hidden_layers = len(kept)
        if s.orig_types: s.cfg.layer_types = [s.orig_types[s.all_layers.index(l)] for l in kept]
    def restore(s):
        super().restore()
        for l, i in zip(s.all_layers, s.orig_idx): l.self_attn.layer_idx = i
        s.cfg.num_hidden_layers = s.orig_n
        if s.orig_types: s.cfg.layer_types = list(s.orig_types)
    def zero(s, p, seed):   # identical to atlas_ext.Dmg.zero
        g = torch.Generator(device=s.model.device).manual_seed(seed)
        with torch.no_grad():
            for n, w in s.params: w.mul_((torch.rand(w.shape, generator=g, device=w.device) >= p).to(w.dtype))
    def neuron(s, p, seed):   # identical to atlas_ext.Dmg.neuron
        g = torch.Generator(device=s.model.device).manual_seed(seed)
        with torch.no_grad():
            for n, w in s.params:
                if n.endswith('mlp.down_proj.weight'): w.mul_((torch.rand(w.shape[1], generator=g, device=w.device) >= p).to(w.dtype)[None, :])
    def qmix(s, p, seed, bits=QBITS, group=64):
        """quantise a random fraction p of the in-layer weight matrices: symmetric round-to-nearest, one scale per group of 64 input weights."""
        q = 2 ** (bits - 1) - 1; names = [n for n, _ in s.params]; chosen = set(random.Random(f'qmix-{seed}').sample(names, int(round(p * len(names)))))
        with torch.no_grad():
            for n, w in s.params:
                if n not in chosen: continue
                x = w.float(); o, i = x.shape; gs = group if i % group == 0 else i
                x = x.view(o, i // gs, gs); sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
                w.copy_(((x / sc).round().clamp(-q, q) * sc).view(o, i).to(w.dtype))
    def apply(s, c):
        k = c['damage']
        if k in ('zero', 'neuron', 'qmix'): s.restore(); getattr(s, k)(c['dose'], c['seed'])
        else:
            super().apply(c)
            if k.startswith('ld_'): s.relayer()

# ---------------------------------------------------------------- evaluation
def batches(lens, extra):
    order = sorted(range(len(lens)), key=lambda i: lens[i]); out = []; cur = []
    for i in order:
        if cur and (len(cur) + 1) * (lens[i] + extra) > BUDGET: out.append(cur); cur = []
        cur.append(i)
    return out + ([cur] if cur else [])
@torch.no_grad()
def logprob_sums(model, tok, pairs):
    """pairs: list of (prompt_ids, cont_ids). Returns the summed log-probability of each continuation."""
    res = [0.0] * len(pairs); lens = [len(p) + len(c) for p, c in pairs]
    for b in batches(lens, 0):
        seqs = [pairs[i][0] + pairs[i][1] for i in b]; m = max(map(len, seqs)); W = max(len(pairs[i][1]) for i in b) + 1
        ids = torch.full((len(b), m), tok.pad_token_id); att = torch.zeros((len(b), m), dtype=torch.long)
        for j, sq in enumerate(seqs): ids[j, m - len(sq):] = torch.tensor(sq); att[j, m - len(sq):] = 1
        pos = (att.cumsum(-1) - 1).clamp_min(0)
        lp = model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda(), logits_to_keep=W, use_cache=False).logits.float().log_softmax(-1)
        for j, i in enumerate(b):
            c = pairs[i][1]; n = len(c); idx = torch.arange(n, device=lp.device)
            res[i] = float(lp[j, W - n - 1 + idx, torch.tensor(c, device=lp.device)].sum())
        del lp
    return res
@torch.no_grad()
def choice_scores(model, tok, prompts, conts):
    """summed log-probability of each continuation in conts[i] (a list of token lists) after prompts[i]; the prompt is run once and its KV cache is shared by its candidates."""
    res = [None] * len(prompts); groups = {}
    for i, c in enumerate(conts): groups.setdefault(len(c), []).append(i)
    for k, idx in groups.items():
        lens = [k * (len(prompts[i]) + max(map(len, conts[i]))) for i in idx]   # the prompt cache is copied k times
        for bb in batches(lens, 0):
            b = [idx[j] for j in bb]; m = max(len(prompts[i]) for i in b)
            ids = torch.full((len(b), m), tok.pad_token_id); att = torch.zeros((len(b), m), dtype=torch.long)
            for j, i in enumerate(b): ids[j, m - len(prompts[i]):] = torch.tensor(prompts[i]); att[j, m - len(prompts[i]):] = 1
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=(att.cumsum(-1) - 1).clamp_min(0).cuda(), use_cache=True, logits_to_keep=1)
            last = out.logits[:, -1].float().log_softmax(-1); pkv = out.past_key_values; pkv.batch_repeat_interleave(k); del out
            cs = [c for i in b for c in conts[i]]; L = max(map(len, cs)) - 1
            x = torch.full((len(cs), L), tok.pad_token_id); cm = torch.zeros((len(cs), L), dtype=torch.long)
            for r, c in enumerate(cs):
                if len(c) > 1: x[r, :len(c) - 1] = torch.tensor(c[:-1]); cm[r, :len(c) - 1] = 1
            a2 = torch.cat([att.repeat_interleave(k, 0), cm], 1); pos = att.sum(1).repeat_interleave(k)[:, None] + torch.arange(L)[None, :]
            lg = model(input_ids=x.cuda(), attention_mask=a2.cuda(), position_ids=pos.cuda(), past_key_values=pkv, use_cache=True).logits
            lp = lg.float().log_softmax(-1) if lg.numel() < 2e8 else torch.cat([z.float().log_softmax(-1) for z in lg.split(64)]); del lg
            for r, c in enumerate(cs):
                v = float(last[r // k, c[0]]) + (float(lp[r, torch.arange(len(c) - 1, device=lp.device), torch.tensor(c[1:], device=lp.device)].sum()) if len(c) > 1 else 0.0)
                i = b[r // k]; res[i] = (res[i] or []) + [v]
            del lp, pkv
    return res
@torch.no_grad()
def generate(model, tok, prompts, mx):
    enc = [tok.encode(p, add_special_tokens=False) for p in prompts]; outs = [''] * len(prompts)
    for b in batches([len(e) for e in enc], mx):
        e = tok([prompts[i] for i in b], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        o = model.generate(**e, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(b, tok.batch_decode(o[:, e['input_ids'].shape[1]:], skip_special_tokens=True)): outs[i] = t
    return outs
def chat(tok, b): return tok.apply_chat_template([{'role': 'user', 'content': b}], tokenize=False, add_generation_prompt=True)
def evaluate(model, tok, end, items, split, tasks=TASKS, formats=FORMATS, cot=False):
    res = {}; jobs = [(t, i, it) for t in tasks for i, it in enumerate(items[t][split])]
    P = {(t, i): chat(tok, body(it, 'BARE')) for t, i, it in jobs}; Pid = {k: tok.encode(v, add_special_tokens=False) for k, v in P.items()}
    if 'CHOICE' in formats:   # forced choice over candidates, continuation = candidate + turn end
        sc_all = choice_scores(model, tok, [Pid[(t, i)] for t, i, it in jobs], [[tok.encode(c, add_special_tokens=False) + [end] for c in it['cands']] for t, i, it in jobs]); k = 0
        for t in tasks:
            sc = sc_all[k:k + len(items[t][split])]; k += len(sc)
            res[f'{t}_CHOICE'] = {'correct': [int(int(np.argmax(x)) == 0) for x in sc], 'scores': [[round(v, 3) for v in x] for x in sc], 'chance': chance(items[t][split][0], 'CHOICE')}
    for f in [x for x in formats if x != 'CHOICE']:
        pr = {(t, i): chat(tok, body(it, f)) for t, i, it in jobs}; keys = list(pr)
        outs = dict(zip(keys, generate(model, tok, [pr[k] for k in keys], GEN_MX[f])))
        for t in tasks:
            its = items[t][split]; J = [judge(it, f, outs[(t, i)]) for i, it in enumerate(its)]
            res[f'{t}_{f}'] = {'correct': [j[0] for j in J], 'format_ok': [j[1] for j in J], 'content_any': [j[2] for j in J], 'outputs': [outs[(t, i)] for i in range(len(its))],
                               'chance': chance(its[0], f)}
            if f == 'BARE' and f'{t}_CHOICE' in res: res[f'{t}_BARE']['gold_lp'] = [x[0] for x in res[f'{t}_CHOICE']['scores']]   # same prompt, same continuation (answer + turn end)
    if cot:
        its = items['GSM8K_COT'][split]; ps = [chat(tok, it['q'] + '\nThink step by step, briefly, then give the final answer in the form "The answer is N."') for it in its]
        outs = generate(model, tok, ps, 384); res['GSM8K_COT'] = {'correct': [judge_cot(o, it['ans']) for o, it in zip(outs, its)], 'outputs': outs, 'chance': 0.0}
    for v in res.values(): v['acc'] = float(np.mean(v['correct']))
    return res

# ---------------------------------------------------------------- damage probe (KL to the intact model)
class Probe:
    def __init__(s, model, tok, texts):
        s.ids = [x for x in (tok.encode(t, add_special_tokens=True)[:48] for t in texts) if len(x) == 48]; s.ref = s.logp(model)
    @torch.no_grad()
    def logp(s, model):
        out = []
        for i in range(0, len(s.ids), 16):
            x = torch.tensor(s.ids[i:i + 16]).cuda(); out.append(model(input_ids=x, use_cache=False).logits.float().log_softmax(-1).half().cpu())
        return out
    @torch.no_grad()
    def measure(s, model):
        kl = []; nll = []
        for (i, ref), cur in zip(enumerate(s.ref), s.logp(model)):
            r = ref.cuda().float(); c = cur.cuda().float(); kl.append(float((r.exp() * (r - c)).sum(-1)[:, :-1].mean()))
            x = torch.tensor(s.ids[i * 16:(i + 1) * 16]).cuda(); nll.append(float(-c[:, :-1].gather(-1, x[:, 1:, None]).mean()))
        return {'kl': float(np.mean(kl)), 'nll': float(np.mean(nll))}

# ---------------------------------------------------------------- calibration, conditions
def calib_metric(ev, base):
    v = [(ev[m]['acc'] - ev[m]['chance']) / (base[m]['acc'] - base[m]['chance']) for m in base if base[m]['acc'] - base[m]['chance'] >= 0.3]
    return float(np.mean(v))
def calibrate(dm, model, tok, end, items, k, mdir):
    f = f'{mdir}/lock_{k}.json'
    if os.path.exists(f): return json.load(open(f))['lock']
    dm.restore(); base = evaluate(model, tok, end, items, 'calib', formats=['BARE']); curve = []
    for x in GRID[k]:
        if k.startswith('ld_') and x > len(dm.all_layers) - 3: break
        vals = []
        for sd in (CALIB_SEEDS if k in STOCH else (0,)):
            dm.apply({'damage': k, 'dose': x, 'seed': sd}); vals.append(calib_metric(evaluate(model, tok, end, items, 'calib', formats=['BARE']), base))
        curve.append((x, float(np.mean(vals)))); print('calib', k, x, round(curve[-1][1], 3), flush=True)
        if curve[-1][1] < 0.05 and len(curve) >= 3: break
    xs = np.array([0.0] + [c[0] for c in curve]); ys = np.minimum.accumulate(np.array([1.0] + [c[1] for c in curve]))
    if k.startswith('ld_'):
        top = next((int(x) for x, y in zip(xs[1:], ys[1:]) if y < 0.08), int(xs[-1])); lock = [n for n in LAYER_DOSES if n <= top]
    else:
        lock = sorted({round(float(np.interp(-t, -ys, xs)), 4) for t in TARGETS if t >= ys[-1]}); lock = [x for x in lock if x > 0]
    dm.restore(); json.dump({'curve': curve, 'lock': lock, 'qbits': QBITS if k == 'qmix' else None}, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f); return lock
def cond_file(mdir, c): return f"{mdir}/{c['damage']}_{c['dose']}_{c.get('seed', 0)}.json.gz"
def run_cond(ctx, c, mdir):
    f = cond_file(mdir, c)
    if os.path.exists(f): return
    model, tok, end, dm, items, probe, meta = ctx; t = time.time(); dm.apply(c)
    pr = probe.measure(model); ev = evaluate(model, tok, end, items, 'eval', cot=(c.get('seed', 0) in (0, 1)))
    out = {'cond': c, 'n_layers': len(dm.model.model.layers), 'probe': pr, 'res': ev, 'secs': round(time.time() - t), **meta}
    with gzip.open(f + '.tmp', 'wt') as fh: json.dump(out, fh, ensure_ascii=False)
    os.replace(f + '.tmp', f); print(meta['model_key'], c, 'probe', {k: round(v, 4) for k, v in pr.items()}, 'meanacc', round(float(np.mean([v['acc'] for v in ev.values()])), 3), out['secs'], 's', flush=True)
def restore_check(ctx):
    model, tok, end, dm, items, probe, meta = ctx; dm.restore(); kl = probe.measure(model)['kl']
    if kl > KL_MAX: raise RuntimeError(f'restore check failed: KL {kl}')
def lock_usable(mdir):
    with gzip.open(f'{mdir}/none_0_0.json.gz', 'rt') as fh: ev = json.load(fh)['res']
    usable = {m: v['acc'] for m, v in ev.items() if m != 'GSM8K_COT' and (v['acc'] - v['chance']) / (1 - v['chance']) >= USABLE}
    tabled = [t for t in TASKS if sum(f'{t}_{f}' in usable for f in FORMATS) >= 3]
    json.dump({'rule': f'(acc-chance)/(1-chance) >= {USABLE}', 'usable': usable, 'tabled_tasks': tabled}, open(f'{mdir}/usable.json', 'w'), indent=1)

# ---------------------------------------------------------------- worker
SKIP = {('q14', 'qmix'), ('gemma', 'qmix')}   # run-time change 1 (2026-10-22): no qmix arm for the two largest models
def jobs_for(key):
    js = [(key, 'intact', 0)]
    for k in DAMAGES:
        if (key, k) not in SKIP: js += [(key, k, s) for s in (MAIN_SEEDS if k in STOCH else (0,))]
    return js
def try_claim(root, j):
    try: os.makedirs(f'{root}/claims/{j[0]}__{j[1]}__{j[2]}'); return True
    except FileExistsError: return False
def wait_for(path, what):
    while not os.path.exists(path): print('waiting for', what, flush=True); time.sleep(60)
def worker(items_dir, root, keys, max_gb, name):
    items, man = load_items(items_dir); ctx = None; cur = None; nfail = 0
    js = [j for k in sorted(keys, key=lambda k: -MODELS[k][1]) for j in jobs_for(k) if MODELS[k][1] <= max_gb]
    while True:
        open_js = [j for j in js if not os.path.exists(f'{root}/claims/{j[0]}__{j[1]}__{j[2]}')]
        if not open_js: print(name, 'no jobs left', flush=True); return
        ready = [j for j in open_js if j[1] == 'intact' or os.path.exists(f'{root}/{j[0]}/usable.json')]
        if not ready: time.sleep(60); continue
        ready.sort(key=lambda j: (j[0] != cur, j[1] != 'intact'))
        j = ready[0]
        if not try_claim(root, j): continue
        key, k, seed = j; mdir = f'{root}/{key}'; os.makedirs(mdir, exist_ok=True); calib_mine = False
        try:
            if key != cur:   # run-time change 2: drop every reference to the old model before loading the next one
                ctx = None; cur = None; model = tok = dm = probe = None; gc.collect(); torch.cuda.empty_cache(); tok, model, end = load_model(key); dm = Dmg(model); probe = Probe(model, tok, items['PROBE']['texts'])
                meta = {'model_key': key, 'model': MODELS[key][0], 'items_sha': man, 'torch': torch.__version__, 'transformers': transformers.__version__, 'host': os.uname().nodename, 'worker': name, 'qbits': QBITS}
                ctx = (model, tok, end, dm, items, probe, meta); cur = key
            model, tok, end, dm, items, probe, meta = ctx; restore_check(ctx)
            if k == 'intact':
                run_cond(ctx, {'damage': 'none', 'dose': 0}, mdir); lock_usable(mdir)
            else:
                lf = f'{mdir}/lock_{k}.json'
                if not os.path.exists(lf):
                    if try_claim(root, (key, 'calib-' + k, 0)): calib_mine = True; calibrate(dm, model, tok, end, items, k, mdir)
                    else: wait_for(lf, f'{key} {k} calibration')
                for x in json.load(open(lf))['lock']: run_cond(ctx, {'damage': k, 'dose': x, 'seed': seed}, mdir)
            restore_check(ctx); open(f'{root}/claims/{key}__{k}__{seed}/done', 'w').write(name); nfail = 0
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True); os.makedirs(f'{root}/failed_log', exist_ok=True)
            open(f'{root}/failed_log/{key}__{k}__{seed}__{name}__{int(time.time())}.txt', 'w').write(tb)
            ctx = None; cur = None; model = tok = dm = probe = None; gc.collect(); torch.cuda.empty_cache(); nfail += 1
            if calib_mine and not os.path.exists(f'{mdir}/lock_{k}.json'): shutil.rmtree(f'{root}/claims/{key}__calib-{k}__0', ignore_errors=True)   # let another worker calibrate
            if 'out of memory' in tb:   # run-time change 2: put the job back (finished conditions are kept and skipped), back off, give up after 3 in a row
                shutil.rmtree(f'{root}/claims/{key}__{k}__{seed}', ignore_errors=True); time.sleep(300)
            else: open(f'{root}/claims/{key}__{k}__{seed}/FAILED', 'w').write(tb)
            if nfail >= 3: print(name, 'three failures in a row, exiting', flush=True); return

# ---------------------------------------------------------------- smoke
def smoke(key, items_dir, root):
    items, man = load_items(items_dir); os.makedirs(f'{root}/smoke', exist_ok=True); t0 = time.time()
    tok, model, end = load_model(key); dm = Dmg(model); probe = Probe(model, tok, items['PROBE']['texts']); print('MODEL', key, 'layers', len(dm.all_layers), 'END', end, repr(tok.decode([end])), 'load', round(time.time() - t0), 's', flush=True)
    torch.cuda.reset_peak_memory_stats(); t = time.time(); ev = evaluate(model, tok, end, items, 'smoke', cot=True); te = time.time() - t
    print('INTACT smoke eval', round(te), 's for', sum(len(items[x]['smoke']) for x in TASKS), 'items x 4 formats; peak GB', round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
    for m, v in ev.items(): print(f'  {m:22s} acc {v["acc"]:.2f}  chance {v["chance"]:.2f}' + (f'  format {np.mean(v["format_ok"]):.2f}  content_any {np.mean(v["content_any"]):.2f}' if 'format_ok' in v else ''))
    samples = {m: [(o, c) for o, c in zip(v['outputs'][:5], v['correct'][:5])] for m, v in ev.items() if 'outputs' in v}
    base = {m: v for m, v in ev.items() if m.endswith('_BARE')}; out = {'intact': {m: v['acc'] for m, v in ev.items()}, 'samples': samples, 'eval_secs': te, 'damaged': {}}
    for c in ({'damage': 'noise', 'dose': 0.1, 'seed': 1}, {'damage': 'zero', 'dose': 0.1, 'seed': 1}, {'damage': 'neuron', 'dose': 0.3, 'seed': 1}, {'damage': 'ld_rand', 'dose': 2, 'seed': 1},
              {'damage': 'ld_late', 'dose': 2}, {'damage': 'qmix', 'dose': 1.0, 'seed': 1}, {'damage': 'qmix', 'dose': 0.5, 'seed': 1}):
        dm.apply(dict(c)); pr = probe.measure(model); e = evaluate(model, tok, end, items, 'smoke', formats=['BARE'])
        out['damaged'][json.dumps(c)] = {'probe': pr, 'bare_metric': calib_metric(e, base)}; print('DAMAGED', c, 'probe', {k: round(v, 4) for k, v in pr.items()}, 'BARE metric', round(out['damaged'][json.dumps(c)]['bare_metric'], 3), flush=True)
    dm.restore(); print('RESTORE KL', probe.measure(model)['kl'])
    json.dump(out, open(f'{root}/smoke/{key}.json', 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    if sys.argv[1] == 'smoke': smoke(sys.argv[2], sys.argv[3], sys.argv[4])
    else: worker(sys.argv[2], sys.argv[3], sys.argv[4].split(','), float(sys.argv[5]), sys.argv[6])
