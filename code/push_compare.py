"""Exploratory pilot: does intact-model instruction push predict observed fragility?  python3 push_compare.py <push_dir> <model> ...
Observed log n: mean over the four random damages (analysis10_8models.json, as analyze11.observed). Only intact-correct items enter the push mean.
"""
import sys, os, gzip, json
import numpy as np
from scipy.stats import spearmanr

A10 = json.load(open('analysis10_out/analysis10_8models.json')); UNT = ['noise', 'zero', 'neuron', 'ld_rand']; F3 = ['BARE', 'JSON', 'TWICE']
ROOTS = ['runs_v8/server_a', 'runs_v8/server_b']
def pear(a, b): return float(np.corrcoef(a, b)[0, 1])
def observed(m):
    v = {}
    for k in UNT:
        for mm, x in (A10['exponents'][m].get(k) or {}).items(): v.setdefault(mm, []).append(np.log(x))
    return {mm: float(np.mean(x)) for mm, x in v.items() if len(x) >= 2}

for m in sys.argv[2:]:
    P = json.load(gzip.open(f'{sys.argv[1]}/{m}.json.gz', 'rt'))['res']; d = next(f'{r}/{m}' for r in ROOTS if os.path.isdir(f'{r}/{m}'))
    base = json.load(gzip.open(f'{d}/none_0_0.json.gz', 'rt'))['res']; obs = observed(m)
    rows = []
    for ms, p in P.items():
        if ms not in obs: continue
        ok = np.array(base[ms]['correct']) == 1
        if ok.sum() < 10: continue
        w, o, L = (np.array(p[k])[ok] for k in ('with', 'without', 'len'))
        rows.append((ms, ms.rsplit('_', 1)[1], obs[ms], float(np.mean(w - o)), float(np.mean(-o)), float(np.mean((w - o) / L)), float(np.mean(-w))))
    names = ['push', 'unnatural(-lp_without)', 'push_per_token', 'own(-lp_with)']
    y = np.array([r[2] for r in rows]); print(f'== {m}: {len(rows)} measures (BARE/JSON/TWICE)')
    for j, nm in enumerate(names):
        x = np.array([r[3 + j] for r in rows]); within = []
        for f in F3:
            s = [i for i, r in enumerate(rows) if r[1] == f]
            if len(s) >= 6: within.append(pear(x[s], y[s]))
        print(f'  {nm:24s} all r={pear(x, y):+.3f} rho={spearmanr(x, y)[0]:+.3f} | within-format r ' + ' '.join(f'{f}:{v:+.2f}' for f, v in zip(F3, within)))
    print('  format means  obs logn:', {f: round(float(np.mean([r[2] for r in rows if r[1] == f])), 2) for f in F3},
          ' push:', {f: round(float(np.mean([r[3] for r in rows if r[1] == f])), 1) for f in F3},
          ' unnatural:', {f: round(float(np.mean([r[4] for r in rows if r[1] == f])), 1) for f in F3})
