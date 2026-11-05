"""Addendum-9 smoke check 3: Qwen2.5-32B on two GPUs, timing of an intact evaluation and of one compression, and restore."""
import time, json, torch
import atlas_full as A, atlas_comp as AC
from atlas_comp import load_model, Dmg
AC.C4 = 'data/c4-validation.00000.json.gz'
t = time.time(); tok, model, end = load_model('q32'); print('load', round(time.time() - t), 's', flush=True)
items, _ = A.load_items('items'); dm = Dmg(model); probe = A.Probe(model, tok, items['PROBE']['texts'])
n_smoke = sum(len(items[x]['smoke']) for x in A.TASKS); n_eval = sum(len(items[x]['eval']) for x in A.TASKS)
t = time.time(); ev = A.evaluate(model, tok, end, items, 'smoke'); te = time.time() - t
print('smoke eval', round(te), 's for', n_smoke, 'items; projected full condition', round(te * n_eval / n_smoke / 60), 'min', flush=True)
for m, v in ev.items(): print(f'  {m:22s} acc {v["acc"]:.2f}')
t = time.time(); dm.apply({'damage': 'wanda', 'dose': 0.5, 'seed': 0}); print('wanda 0.5', round(time.time() - t), 's; KL', round(probe.measure(model)['kl'], 4), flush=True)
t = time.time(); dm.apply({'damage': 'noise', 'dose': 0.2, 'seed': 1}); print('noise 0.2 (incl. disk restore)', round(time.time() - t), 's; KL', round(probe.measure(model)['kl'], 4), flush=True)
t = time.time(); dm.restore(); print('restore', round(time.time() - t), 's; KL', probe.measure(model)['kl'], flush=True)
