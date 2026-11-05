"""Damage-atlas pilot. See PROTOCOL.md.
  python3 atlas.py smoke                     # tiny run, intact model only
  python3 atlas.py calib  <outdir>           # calibration items, wide dose grid
  python3 atlas.py main   <outdir> <shard> <nshards>   # locked doses, evaluation items
"""
import sys, os, json, random, re, math, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get('ATLAS_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
LETTERS = ['A', 'B', 'C', 'D']
CAPITALS = [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),("Portugal","Lisbon"),("Japan","Tokyo"),("China","Beijing"),("India","New Delhi"),("Russia","Moscow"),("Canada","Ottawa"),("Australia","Canberra"),("Brazil","Brasilia"),("Argentina","Buenos Aires"),("Mexico","Mexico City"),("Egypt","Cairo"),("Kenya","Nairobi"),("Nigeria","Abuja"),("Turkey","Ankara"),("Greece","Athens"),("Sweden","Stockholm"),("Norway","Oslo"),("Finland","Helsinki"),("Denmark","Copenhagen"),("Poland","Warsaw"),("Austria","Vienna"),("Hungary","Budapest"),("Ireland","Dublin"),("Belgium","Brussels"),("Netherlands","Amsterdam"),("Thailand","Bangkok"),("Vietnam","Hanoi"),("Indonesia","Jakarta"),("Philippines","Manila"),("South Korea","Seoul"),("Iran","Tehran"),("Iraq","Baghdad"),("Saudi Arabia","Riyadh"),("Peru","Lima"),("Chile","Santiago"),("Colombia","Bogota"),("Cuba","Havana"),("Ukraine","Kyiv"),("Czech Republic","Prague"),("Romania","Bucharest"),("Bulgaria","Sofia"),("Serbia","Belgrade"),("Croatia","Zagreb"),("Morocco","Rabat"),("Ethiopia","Addis Ababa"),("Ghana","Accra"),("Pakistan","Islamabad"),("Bangladesh","Dhaka"),("Nepal","Kathmandu"),("Afghanistan","Kabul"),("Israel","Jerusalem"),("Jordan","Amman"),("Lebanon","Beirut"),("Syria","Damascus"),("Venezuela","Caracas"),("Ecuador","Quito"),("Uruguay","Montevideo"),("Iceland","Reykjavik"),("New Zealand","Wellington"),("Malaysia","Kuala Lumpur"),("Mongolia","Ulaanbaatar"),("Kazakhstan","Astana"),("Algeria","Algiers"),("Tunisia","Tunis"),("Libya","Tripoli"),("Sudan","Khartoum"),("Uganda","Kampala"),("Zimbabwe","Harare"),("Senegal","Dakar"),("Belarus","Minsk"),("Latvia","Riga"),("Lithuania","Vilnius"),("Estonia","Tallinn"),("Slovakia","Bratislava"),("Qatar","Doha"),("Taiwan","Taipei")]
SING = ["key","label","picture","report","letter","sign","note","message","list","door","book","map","photo","bill","gift","answer"]
PLUR = ["cabinets","boxes","teachers","doctors","managers","students","windows","gardens","children","neighbors","drivers","tables","shelves","writers","farmers","pilots"]
PREP = ["to","near","for","from","beside","behind","with","about"]
ADJ = ["old","new","ready","here","missing","broken","important","on the table"]
TASKS = ['ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'AGREE']

def gen_items(task, n, seed):
    r = random.Random(f'{task}-{seed}'); out = []
    for _ in range(n):
        if task.startswith('ARITH'):
            k = int(task[-1]); xs = [r.randint(11, 49) for _ in range(k + 1)]; q, ans = ' + '.join(map(str, xs)), sum(xs)
            ds = set()
            while len(ds) < 3:
                x = ans + r.choice([-20, -11, -10, -9, -2, -1, 1, 2, 9, 10, 11, 20])
                if x != ans and x > 0: ds.add(x)
            out.append({'q': f"What is {q}?", 'ans': str(ans), 'dist': [str(x) for x in ds], 'kind': 'number'})
        elif task == 'FACT':
            c, cap = r.choice(CAPITALS); ds = r.sample([x[1] for x in CAPITALS if x[1] != cap], 3)
            out.append({'q': f"What is the capital of {c}?", 'ans': cap, 'dist': ds, 'kind': 'text'})
        elif task == 'RETR':
            keys = set()
            while len(keys) < 12: keys.add(''.join(r.choice('bcdfghjklmnprstvz') + r.choice('aeiou') for _ in range(3)))
            keys = list(keys); vals = r.sample(range(100, 1000), 12); j = r.randrange(12)
            ctx = '\n'.join(f"{k}: {v}" for k, v in zip(keys, vals))
            ds = r.sample([str(v) for i, v in enumerate(vals) if i != j], 3)
            out.append({'q': f"Here is a list of codes.\n{ctx}\nWhat is the code for {keys[j]}?", 'ans': str(vals[j]), 'dist': ds, 'kind': 'number'})
        elif task == 'AGREE':
            s, p, pr, adj = r.choice(SING), r.choice(PLUR), r.choice(PREP), r.choice(ADJ)
            if r.random() < 0.5: subj, ans = f"The {s} {pr} the {p}", 'is'
            else: subj, ans = f"The {p} {pr} the {s}", 'are'
            out.append({'q': f"Fill in the blank with the grammatically correct word: \"{subj} ___ {adj}.\"", 'ans': ans, 'dist': ['are' if ans == 'is' else 'is'], 'kind': 'text'})
    return out

def chance(task): return 0.5 if task == 'AGREE' else 0.25

def build_prompts(tok, task, items, fmt, seed):
    r = random.Random(f'opt-{task}-{seed}'); prompts = []; gold = []
    for it in items:
        if fmt == 'LETTER':
            opts = [it['ans']] + it['dist']; r.shuffle(opts)
            body = it['q'] + '\n' + '\n'.join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer with the letter only."
            gold.append(opts.index(it['ans']))
        else:
            tail = "Answer with just the number." if it['kind'] == 'number' else ("Answer with just the word 'is' or 'are'." if task == 'AGREE' else "Answer with just the name.")
            body = it['q'] + '\n' + tail; gold.append(it['ans'])
        prompts.append(tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True))
    return prompts, gold

@torch.no_grad()
def evaluate(model, tok, sets, bs=50):
    """sets: {task: items}. returns {task_fmt: {'acc':..., 'correct':[0/1...]}}"""
    dev = model.device; res = {}
    lid = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    for task, (items, seed) in sets.items():
        for fmt in ['GEN', 'CHOICE', 'LETTER']:
            prompts, gold = build_prompts(tok, task, items, 'GEN' if fmt == 'CHOICE' else fmt, seed); corr = []
            if fmt == 'CHOICE':
                # forced choice by likelihood of each candidate answer string (same prompt as GEN)
                end = [tok.eos_token_id]
                for i in range(0, len(prompts), 16):
                    seqs = []; spans = []
                    for p, it in zip(prompts[i:i + 16], items[i:i + 16]):
                        pid = tok.encode(p, add_special_tokens=False)
                        for cnd in [it['ans']] + it['dist']:
                            cid = tok.encode(cnd, add_special_tokens=False) + end; seqs.append(pid + cid); spans.append(len(cid))
                    m = max(map(len, seqs)); ids = torch.full((len(seqs), m), tok.pad_token_id); att = torch.zeros((len(seqs), m), dtype=torch.long)
                    for j, sq in enumerate(seqs): ids[j, m - len(sq):] = torch.tensor(sq); att[j, m - len(sq):] = 1
                    W = max(spans) + 1      # only the last W positions are needed (sequences are left-padded)
                    lp = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, -W:, :].float().log_softmax(-1)
                    sc = [lp[j, W - n - 1 + t, seqs[j][len(seqs[j]) - n + t]].item() for j, n in enumerate(spans) for t in range(n)]
                    del lp
                    k = 0; tot = []
                    for n in spans: tot.append(sum(sc[k:k + n])); k += n
                    nc = len([items[i]['ans']] + items[i]['dist'])
                    for a in range(0, len(tot), nc): corr.append(int(max(range(nc), key=lambda z: tot[a + z]) == 0))
                res[f'{task}_{fmt}'] = {'acc': sum(corr) / len(corr), 'correct': corr}; continue
            for i in range(0, len(prompts), bs):
                enc = tok(prompts[i:i + bs], return_tensors='pt', padding=True).to(dev)
                if fmt == 'LETTER':
                    lg = model(**enc).logits[:, -1, :].float()
                    nopt = 2 if task == 'AGREE' else 4
                    pred = lg[:, lid[:nopt]].argmax(-1).tolist()
                    corr += [int(p == g) for p, g in zip(pred, gold[i:i + bs])]
                else:
                    out = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
                    txt = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
                    for t, g, it in zip(txt, gold[i:i + bs], items[i:i + bs]):
                        t = t.strip()
                        if it['kind'] == 'number':
                            m = re.search(r'-?\d[\d,]*', t); ok = bool(m) and m.group(0).replace(',', '') == g
                        elif task == 'AGREE':
                            m = re.search(r"\b(is|are)\b", t.lower()); ok = bool(m) and m.group(1) == g
                        else: ok = t.lower().strip(" .'\"").startswith(g.lower())
                        corr.append(int(ok))
            res[f'{task}_{fmt}'] = {'acc': sum(corr) / len(corr), 'correct': corr}
    return res

def target_params(model): return [(n, p) for n, p in model.named_parameters() if p.dim() == 2 and '.layers.' in n]

class Damager:
    def __init__(s, model):
        s.model = model; s.params = target_params(model); s.orig = {n: p.detach().to('cpu', copy=True) for n, p in s.params}
        s.layers = model.model.layers; s.all_layers = list(s.layers)
    def restore(s):
        with torch.no_grad():
            for n, p in s.params: p.copy_(s.orig[n].to(p.device))
        s.model.model.layers = torch.nn.ModuleList(s.all_layers)
    def noise(s, r, seed):
        g = torch.Generator(device=s.model.device).manual_seed(seed)
        with torch.no_grad():
            for n, p in s.params:
                w = p.float(); p.copy_((w + torch.randn(w.shape, generator=g, device=w.device) * (r * w.std())).to(p.dtype))
    def quant(s, q):
        """round-to-nearest, symmetric, per-output-row uniform quantisation with q positive levels (b bits <=> q = 2**(b-1)-1)."""
        with torch.no_grad():
            for n, p in s.params:
                w = p.float(); sc = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / q
                p.copy_(((w / sc).round().clamp(-q, q) * sc).to(p.dtype))
    def layerdrop(s, nd):
        """remove nd consecutive layers ending just before the last layer (last layer kept)."""
        L = len(s.all_layers); keep = s.all_layers[:L - 1 - nd] + [s.all_layers[-1]]
        s.model.model.layers = torch.nn.ModuleList(keep)
    def layerdrop_at(s, nd, where, seed=0):
        """addendum 1: remove nd layers at a given position. first and last layer are always kept."""
        L = len(s.all_layers)
        if where == 'early': drop = set(range(1, 1 + nd))
        elif where == 'mid': st = (L - nd) // 2; drop = set(range(st, st + nd))
        elif where == 'late': drop = set(range(L - 1 - nd, L - 1))
        else: drop = set(random.Random(f'ld-{seed}-{nd}').sample(range(1, L - 1), nd))
        s.model.model.layers = torch.nn.ModuleList([l for i, l in enumerate(s.all_layers) if i not in drop]); return sorted(drop)
    def apply(s, cond):
        s.restore(); k = cond['damage']
        if k.startswith('ld_'): cond['dropped'] = s.layerdrop_at(cond['dose'], k[3:], cond.get('seed', 0)); return
        if k == 'noise': s.noise(cond['dose'], cond['seed'])
        elif k == 'quant': s.quant(cond['dose'])          # dose = q positive levels (amendment 1)
        elif k == 'quantbits': s.quant(2 ** (cond['dose'] - 1) - 1)
        elif k == 'layerdrop': s.layerdrop(cond['dose'])

def load():
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('cuda').eval()
    return tok, model

CALIB = {'noise': [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30], 'quantbits': [8, 7, 6, 5, 4, 3, 2], 'layerdrop': [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]}

def run(conds, sets, outdir, tag):
    tok, model = load(); dm = Damager(model); os.makedirs(outdir, exist_ok=True)
    for c in conds:
        f = f"{outdir}/{tag}_{c['damage']}_{c['dose']}_{c.get('seed', 0)}.json"
        if os.path.exists(f): continue
        t = time.time(); dm.apply(c); ev = evaluate(model, tok, sets)
        json.dump({'cond': c, 'model': MODEL, 'n_layers': len(dm.all_layers), 'eval': ev}, open(f, 'w'))
        print(tag, c, {k: round(v['acc'], 2) for k, v in ev.items()}, round(time.time() - t), 's', flush=True)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'smoke':
        tok, model = load(); sets = {t: (gen_items(t, 20, 'smoke'), 'smoke') for t in TASKS}
        ev = evaluate(model, tok, sets); print({k: v['acc'] for k, v in ev.items()})
        dm = Damager(model); dm.apply({'damage': 'layerdrop', 'dose': 2}); print('layerdrop2', {k: v['acc'] for k, v in evaluate(model, tok, sets).items()})
        dm.apply({'damage': 'quant', 'dose': 4}); print('quant4', {k: v['acc'] for k, v in evaluate(model, tok, sets).items()})
        dm.apply({'damage': 'noise', 'dose': 0.05, 'seed': 1}); print('noise.05', {k: v['acc'] for k, v in evaluate(model, tok, sets).items()})
        dm.restore(); print('restored', {k: v['acc'] for k, v in evaluate(model, tok, sets).items()})
    elif mode == 'calib':
        sets = {t: (gen_items(t, 60, 'calib'), 'calib') for t in TASKS}
        conds = [{'damage': 'none', 'dose': 0}] + [{'damage': d, 'dose': x, 'seed': 7} for d in CALIB for x in CALIB[d]]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'calib')
    elif mode == 'calib2':
        sets = {t: (gen_items(t, 60, 'calib'), 'calib') for t in TASKS}
        conds = [{'damage': 'noise', 'dose': x, 'seed': 7} for x in (0.35, 0.40, 0.50, 0.60)] + [{'damage': 'quant', 'dose': q} for q in (6, 5, 4)] + [{'damage': 'layerdrop', 'dose': 5}]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'calib')
    elif mode == 'calibC':
        sets = {t: (gen_items(t, 60, 'calib'), 'calib') for t in TASKS}
        conds = [{'damage': 'none', 'dose': 0}] + [{'damage': 'noise', 'dose': x, 'seed': sd} for x in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50) for sd in (11, 12, 13, 14)]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'calib')
    elif mode == 'mainC':
        doses = json.load(open(os.environ['ATLAS_LOCK']))
        sets = {t: (gen_items(t, 200, 'eval'), 'eval') for t in TASKS}
        conds = [{'damage': 'noise', 'dose': x, 'seed': sd} for x in doses['noise'] for sd in (1, 2, 3, 4, 5)]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'mainC')
    elif mode == 'calibB':
        sets = {t: (gen_items(t, 60, 'calib'), 'calib') for t in TASKS}
        conds = [{'damage': 'none', 'dose': 0}] + [{'damage': 'noise', 'dose': x, 'seed': 7} for x in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)] + [{'damage': 'quant', 'dose': q} for q in (31, 15, 10, 8, 7, 6, 5, 4, 3)]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'calib')
    elif mode == 'mainB':
        doses = json.load(open(os.environ['ATLAS_LOCK'])); NS = [1, 2, 3, 4, 5, 6, 8, 10]
        sets = {t: (gen_items(t, 200, 'eval'), 'eval') for t in TASKS}
        conds = [{'damage': 'none', 'dose': 0}] + [{'damage': 'noise', 'dose': x, 'seed': sd} for x in doses['noise'] for sd in (1, 2, 3)] + [{'damage': 'quant', 'dose': q} for q in doses['quant']]
        conds += [{'damage': 'ld_rand', 'dose': n, 'seed': sd} for n in NS for sd in (1, 2, 3)] + [{'damage': f'ld_{w}', 'dose': n} for w in ('mid', 'late') for n in NS]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'mainB')
    elif mode == 'main2':
        NS = [1, 2, 3, 4, 5, 6, 8, 10]
        sets = {t: (gen_items(t, 200, 'eval'), 'eval') for t in TASKS}
        conds = [{'damage': f'ld_{w}', 'dose': n} for w in ('early', 'mid', 'late') for n in NS] + [{'damage': 'ld_rand', 'dose': n, 'seed': sd} for n in NS for sd in (1, 2, 3)]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'main2')
    elif mode == 'main':
        doses = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LOCKED_DOSES.json')))
        sets = {t: (gen_items(t, 200, 'eval'), 'eval') for t in TASKS}
        conds = [{'damage': 'none', 'dose': 0}] + [{'damage': 'noise', 'dose': x, 'seed': s} for x in doses['noise'] for s in (1, 2, 3)] + [{'damage': d, 'dose': x} for d in ('quant', 'layerdrop') for x in doses[d]]
        sh, ns = int(sys.argv[3]), int(sys.argv[4])
        run([c for i, c in enumerate(conds) if i % ns == sh], sets, sys.argv[2], 'main')
