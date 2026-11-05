"""Extension to more models and damage types (PROTOCOL_ADDENDUM_7.md).
  python3 atlas_ext.py smoke
  python3 atlas_ext.py run <outdir> <comma-separated damages>
env: ATLAS_MODEL
"""
import sys, os, json, random, time, torch, numpy as np
import atlas, atlas_bench as B, atlas_fmt as F
from atlas import Damager, load
GRID = {'noise': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5], 'zero': [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6], 'neuron': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'ld_rand': [1, 2, 3, 4, 6, 8, 10, 12, 16], 'ld_late': [1, 2, 3, 4, 6, 8, 10, 12, 16]}
TARGETS = [0.9, 0.75, 0.6, 0.45, 0.3, 0.12]; STOCH = ('noise', 'zero', 'neuron', 'ld_rand')
class Dmg(Damager):
    def zero(s, p, seed):
        g = torch.Generator(device=s.model.device).manual_seed(seed)
        with torch.no_grad():
            for n, w in s.params: w.mul_((torch.rand(w.shape, generator=g, device=w.device) >= p).to(w.dtype))
    def neuron(s, p, seed):
        """zero a random fraction p of MLP hidden units by zeroing the matching columns of each mlp.down_proj weight."""
        g = torch.Generator(device=s.model.device).manual_seed(seed)
        with torch.no_grad():
            for n, w in s.params:
                if n.endswith('mlp.down_proj.weight'): w.mul_((torch.rand(w.shape[1], generator=g, device=w.device) >= p).to(w.dtype)[None, :])
    def apply(s, c):
        k = c['damage']
        if k == 'zero': s.restore(); s.zero(c['dose'], c['seed'])
        elif k == 'neuron': s.restore(); s.neuron(c['dose'], c['seed'])
        else: super().apply(c)
def setup():
    tok, model = load()
    # turn-end token = whatever the chat template puts right after an assistant message (equals tokenizer.eos for Qwen and Mistral, as in earlier batches)
    txt = tok.apply_chat_template([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'ZQX'}], tokenize=False); after = txt.split('ZQX')[-1]
    ids = tok.encode(after, add_special_tokens=False); B.END_ID = ids[0] if ids else tok.eos_token_id
    if os.environ.get('ATLAS_NOCACHE'): model.generation_config.use_cache = False   # Gemma-2: cached generation crashes for some sets of removed layers
    for t in list(B.FORMATS):
        if t in B.OLD: B.FORMATS[t] = ['GEN', 'CHOICE']
        else: del B.FORMATS[t]
    assert any(n.endswith('mlp.down_proj.weight') for n, _ in atlas.target_params(model)), 'no mlp.down_proj found'
    return tok, model, Dmg(model)
def meanc(ev, base):
    v = [(ev[m]['acc'] - ev[m]['chance']) / (base[m]['acc'] - base[m]['chance']) for m in base if base[m]['acc'] - base[m]['chance'] >= 0.3]
    return float(np.mean(v))
def calibrate(dm, model, tok, k, out):
    f = f'{out}/calib_{k}.json'
    if os.path.exists(f): return json.load(open(f))['lock']
    cs = {t: (B.get_items(t, 40, 'calib'), 'calib') for t in B.FORMATS}; dm.restore(); base = B.evaluate(model, tok, cs); curve = []
    for x in GRID[k]:
        if k in ('ld_rand', 'ld_late') and x > len(dm.all_layers) - 3: break
        vals = []
        for sd in ((11, 12) if k in STOCH else (0,)):
            dm.apply({'damage': k, 'dose': x, 'seed': sd}); vals.append(meanc(B.evaluate(model, tok, cs), base))
        curve.append((x, float(np.mean(vals)))); print('calib', k, x, round(curve[-1][1], 3), flush=True)
        if curve[-1][1] < 0.05 and len(curve) >= 3: break
    xs = np.array([0.0] + [c[0] for c in curve]); ys = np.minimum.accumulate(np.array([1.0] + [c[1] for c in curve]))
    if k in ('ld_rand', 'ld_late'):
        top = next((int(x) for x, y in zip(xs[1:], ys[1:]) if y < 0.08), int(xs[-1])); lock = [n for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16) if n <= top]
    else:
        lock = sorted({round(float(np.interp(-t, -ys, xs)), 3) for t in TARGETS if t >= ys[-1]}); lock = [x for x in lock if x > 0]
    dm.restore(); json.dump({'curve': curve, 'lock': lock}, open(f, 'w')); return lock
def run(out, damages):
    tok, model, dm = setup(); os.makedirs(out, exist_ok=True)
    sets = {t: F.items_for(t, 200, 'eval') for t in F.CONTENT + ['COPYLONG']}; ref = {t: (B.get_items(t, 200, 'eval'), 'eval') for t in B.FORMATS}
    def do(c):
        f = f"{out}/ext_{c['damage']}_{c['dose']}_{c.get('seed', 0)}.json"
        if os.path.exists(f): return
        t = time.time(); dm.apply(c); r = F.eval_fmt(model, tok, sets); rf = B.evaluate(model, tok, ref)
        for v in r.values(): v.pop('outputs')
        json.dump({'cond': c, 'model': atlas.MODEL, 'n_layers': len(dm.all_layers), 'fmt': r, 'ref': rf}, open(f, 'w'))
        print('ext', c, 'meanref', round(float(np.mean([v['acc'] for v in rf.values()])), 2), round(time.time() - t), 's', flush=True)
    do({'damage': 'none', 'dose': 0})
    for k in damages:
        lock = calibrate(dm, model, tok, k, out); print('LOCK', k, lock, flush=True)
        for x in lock:
            for sd in ((1, 2, 3) if k in STOCH else (0,)): do({'damage': k, 'dose': x, 'seed': sd})
if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        tok, model, dm = setup(); print('MODEL', atlas.MODEL, 'layers', len(dm.all_layers), 'END_ID', B.END_ID, 'eos', tok.eos_token_id)
        sets = {t: F.items_for(t, 30, 'smoke') for t in F.CONTENT + ['COPYLONG']}; ref = {t: (B.get_items(t, 30, 'smoke'), 'smoke') for t in B.FORMATS}
        r = F.eval_fmt(model, tok, sets); rf = B.evaluate(model, tok, ref)
        print('FORMAT', {k: round(float(np.mean(v['format'])), 2) for k, v in r.items()}); print('CONTENT', {k: round(float(np.mean(v['content'])), 2) for k, v in r.items()}); print('REF', {k: round(v['acc'], 2) for k, v in rf.items()})
        print('sample', repr(r['FACT_F2']['outputs'][0]), repr(r['ARITH1_F0']['outputs'][0]), repr(r['COPYLONG']['outputs'][0]))
        for c in ({'damage': 'zero', 'dose': 0.3, 'seed': 1}, {'damage': 'neuron', 'dose': 0.5, 'seed': 1}, {'damage': 'ld_late', 'dose': 2}):
            dm.apply(c); print(c, {k: round(v['acc'], 2) for k, v in B.evaluate(model, tok, ref).items()})
        dm.restore(); print('restored', {k: round(v['acc'], 2) for k, v in B.evaluate(model, tok, ref).items()})
    else: run(sys.argv[2], sys.argv[3].split(','))
