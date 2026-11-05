"""Runner for PROTOCOL_ADDENDUM_9: importance-guided compression (Wanda, SparseGPT, GPTQ) on the seven models of addendum 8, and the
untargeted damages plus compression on Qwen2.5-32B-Instruct. Evaluation, judging, probes, dose targets and the worker queue are those
of atlas_full.py (imported, not copied); only the damage set, the dose calibration for the new damages and the 32B loading differ.
  python3 atlas_comp.py smoke  <items_dir> <out_root> <c4_path> <wikitext_path>
  python3 atlas_comp.py worker <items_dir> <out_root> <comma-separated model keys> <max_gb> <worker_name>
env: ATLAS_C4 (path to the C4 validation shard), ATLAS_TOK_BUDGET as in atlas_full.py, ATLAS_MAXMEM (per-GPU cap when a model
is split, default 44GiB; one value for all cards or one per visible card, comma-separated),
ATLAS_SPLIT (decoder-layer counts for every visible card but the last, comma-separated; overrides ATLAS_MAXMEM), ATLAS_JOBS (comma-separated damages this worker may take; default all)
"""
import sys, os, json, glob, math, time, gzip
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import atlas_full as A
from atlas import target_params
import compress as C

MODELS = dict(A.MODELS); MODELS['q32'] = ('Qwen/Qwen2.5-32B-Instruct', 66)
COMP = ['wanda', 'sparsegpt', 'gptq']
GRID = {'wanda': [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], 'sparsegpt': [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'gptq': list(C.GPTQ_CFG)}
C4 = os.environ.get('ATLAS_C4', '')
A_load_model = A.load_model

def load_model(key):
    if key != 'q32': return A_load_model(key)
    mid = MODELS[key][0]; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ng = torch.cuda.device_count()   # one 80 GB card holds the whole model; two 48 GB cards split it
    dmap = {'device_map': {'': 0}} if ng == 1 else {'device_map': 'auto', 'max_memory': dict(enumerate((os.environ.get('ATLAS_MAXMEM', '44GiB').split(',') * ng)[:ng]))}
    if os.environ.get('ATLAS_SPLIT'):   # explicit split: layer counts for every card but the last; the rest and the head go on the last card
        ks = [int(x) for x in os.environ['ATLAS_SPLIT'].split(',')]; assert len(ks) == ng - 1, (ks, ng)
        n = AutoConfig.from_pretrained(mid).num_hidden_layers; edges = np.cumsum(ks)
        dm_ = {'model.embed_tokens': 0, 'model.norm': ng - 1, 'model.rotary_emb': ng - 1, 'lm_head': ng - 1}
        dm_.update({f'model.layers.{i}': int(np.searchsorted(edges, i, side='right')) for i in range(n)})
        dmap = {'device_map': dm_}
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, **dmap).eval()
    txt = tok.apply_chat_template([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'ZQX'}], tokenize=False)
    ids = tok.encode(txt.split('ZQX')[-1], add_special_tokens=False); end = ids[0] if ids else tok.eos_token_id
    return tok, model, end

class Dmg(A.Dmg):
    """adds the three compressions; for multi-GPU models, restores from the checkpoint on disk (no CPU copy) and draws random
    masks with one generator per device."""
    def __init__(s, model):
        s.multi = len({p.device for p in model.parameters()}) > 1
        if not s.multi: super().__init__(model)
        else:
            s.model = model; s.params = target_params(model); s.layers = model.model.layers; s.all_layers = list(s.layers)
            s.cfg = model.config; s.orig_n = s.cfg.num_hidden_layers; s.orig_types = list(getattr(s.cfg, 'layer_types', None) or []) or None
            s.orig_idx = [l.self_attn.layer_idx for l in s.all_layers]; s.src = s._index()
        s.calib = None
    def _index(s):
        from huggingface_hub import snapshot_download
        d = snapshot_download(s.cfg._name_or_path, allow_patterns=['*.safetensors', '*.json']); idx = json.load(open(f'{d}/model.safetensors.index.json'))['weight_map']
        return {n: f'{d}/{idx[n]}' for n, _ in s.params}
    def restore(s):
        if not s.multi: return super().restore()
        from safetensors import safe_open
        by_file = {}
        for n, p in s.params: by_file.setdefault(s.src[n], []).append((n, p))
        with torch.no_grad():
            for f, lst in by_file.items():
                with safe_open(f, framework='pt') as fh:
                    for n, p in lst: p.copy_(fh.get_tensor(n).to(p.device))
        s.model.model.layers = torch.nn.ModuleList(s.all_layers)
        for l, i in zip(s.all_layers, s.orig_idx): l.self_attn.layer_idx = i
        s.cfg.num_hidden_layers = s.orig_n
        if s.orig_types: s.cfg.layer_types = list(s.orig_types)
    def _gen(s, seed):
        gens = {}
        def g(dev):
            if dev not in gens: gens[dev] = torch.Generator(device=dev).manual_seed(seed)
            return gens[dev]
        return g
    def noise(s, r, seed):
        if not s.multi: return super().noise(r, seed)
        g = s._gen(seed)
        with torch.no_grad():
            for n, p in s.params: w = p.float(); p.copy_((w + torch.randn(w.shape, generator=g(w.device), device=w.device) * (r * w.std())).to(p.dtype))
    def zero(s, p_, seed):
        if not s.multi: return super().zero(p_, seed)
        g = s._gen(seed)
        with torch.no_grad():
            for n, w in s.params: w.mul_((torch.rand(w.shape, generator=g(w.device), device=w.device) >= p_).to(w.dtype))
    def neuron(s, p_, seed):
        if not s.multi: return super().neuron(p_, seed)
        g = s._gen(seed)
        with torch.no_grad():
            for n, w in s.params:
                if n.endswith('mlp.down_proj.weight'): w.mul_((torch.rand(w.shape[1], generator=g(w.device), device=w.device) >= p_).to(w.dtype)[None, :])
    def apply(s, c):
        k = c['damage']
        if k in COMP:
            s.restore()
            if s.calib is None:
                tok = AutoTokenizer.from_pretrained(s.cfg._name_or_path); s.calib = C.c4_calibration(tok, C4)
            C.compress(s.model, k, c['dose'], s.calib)
        elif k == 'noise' and s.multi: s.restore(); s.noise(c['dose'], c['seed'])
        else: super().apply(c)

def calibrate(dm, model, tok, end, items, k, mdir):
    if k not in COMP: return A_calibrate(dm, model, tok, end, items, k, mdir)
    f = f'{mdir}/lock_{k}.json'
    if os.path.exists(f): return json.load(open(f))['lock']
    dm.restore(); base = A.evaluate(model, tok, end, items, 'calib', formats=['BARE']); curve = []
    for x in GRID[k]:
        dm.apply({'damage': k, 'dose': x, 'seed': 0}); curve.append((x, A.calib_metric(A.evaluate(model, tok, end, items, 'calib', formats=['BARE']), base)))
        print('calib', k, x, round(curve[-1][1], 3), flush=True)
        if curve[-1][1] < 0.05 and len(curve) >= 3: break
    xs = [c[0] for c in curve]; ys = list(np.minimum.accumulate(np.array([c[1] for c in curve])))
    if k == 'gptq':   # discrete configurations: keep those inside [0.08, 0.95]; fill up to 4 with the nearest ones
        inside = [x for x, y in zip(xs, ys) if 0.08 <= y <= 0.95]
        rest = sorted([(min(abs(y - 0.08), abs(y - 0.95)), x) for x, y in zip(xs, ys) if x not in inside])
        lock = sorted(inside + [x for _, x in rest[:max(0, 4 - len(inside))]])
    else:
        X = np.array([0.0] + xs); Y = np.array([1.0] + ys)
        lock = sorted({round(float(np.interp(-t, -Y, X)), 4) for t in A.TARGETS if t >= Y[-1]}); lock = [x for x in lock if x > 0]
    dm.restore(); json.dump({'curve': curve, 'lock': lock}, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f); return lock

def jobs_for(key):
    js = [(key, 'intact', 0)] if key == 'q32' else []   # the seven earlier models reuse their addendum-8 intact results (same code, same items)
    if key == 'q32': js += [(key, k, s) for k in A.UNT for s in A.MAIN_SEEDS]
    js += [(key, k, 0) for k in COMP]
    only = os.environ.get('ATLAS_JOBS')   # split one model's damages between machines that do not share a claims folder
    return [j for j in js if j[1] in only.split(',')] if only else js

A_calibrate = A.calibrate
A.MODELS = MODELS; A.Dmg = Dmg; A.load_model = load_model; A.calibrate = calibrate; A.jobs_for = jobs_for
A.STOCH = tuple(A.STOCH)   # compression is deterministic (seed 0); untargeted damages keep their seeds

# ---------------------------------------------------------------- smoke (protocol checks 1-3)
@torch.no_grad()
def ppl(model, tok, path, n=40, L=512):
    import pyarrow.parquet as pq
    text = '\n\n'.join(pq.read_table(path).column('text').to_pylist()); ids = tok(text, return_tensors='pt').input_ids[0]
    nll = []
    for i in range(n):
        x = ids[i * L:(i + 1) * L][None].to(next(model.parameters()).device)
        nll.append(float(model(x, labels=x, use_cache=False).loss))
    return math.exp(np.mean(nll))

def smoke(items_dir, root, c4, wiki, key='q15'):
    global C4; C4 = c4; os.makedirs(f'{root}/smoke', exist_ok=True)
    tok, model, end = load_model(key); dm = Dmg(model); items, _ = A.load_items(items_dir); probe = A.Probe(model, tok, items['PROBE']['texts'])
    out = {'ppl_intact': ppl(model, tok, wiki)}; print('intact ppl', round(out['ppl_intact'], 3), flush=True)
    for k, d in (('wanda', 0.5), ('sparsegpt', 0.5), ('gptq', 1)):
        t = time.time(); dm.apply({'damage': k, 'dose': d, 'seed': 0}); secs = time.time() - t
        sp = float(np.mean([(p == 0).float().mean().item() for _, p in dm.params])); pp = ppl(model, tok, wiki)
        out[f'{k}_{d}'] = {'ppl': pp, 'rel_increase': pp / out['ppl_intact'] - 1, 'sparsity': sp, 'secs': secs}
        print(k, d, 'ppl', round(pp, 3), 'rel', round(pp / out['ppl_intact'] - 1, 3), 'sparsity', round(sp, 4), 'secs', round(secs), flush=True)
    dm.restore(); out['restore_kl'] = probe.measure(model)['kl']; print('restore KL', out['restore_kl'])
    json.dump(out, open(f'{root}/smoke/{key}_comp.json', 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == 'smoke': smoke(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], *(sys.argv[6:7]))
    else: A.worker(sys.argv[2], sys.argv[3], sys.argv[4].split(','), float(sys.argv[5]), sys.argv[6])
