"""Locked analysis for PROTOCOL_ADDENDUM_3 (Mistral noise redo). Written before any mainC result was opened."""
import json, glob, itertools
import numpy as np
A3 = json.load(open('mis_main/analysis3.json')); Q1 = json.load(open('main/analysis.json')); usable = A3['usable']
base = {m: np.array(v['correct']) for m, v in json.load(open(glob.glob('mis_main/mainB_none_*.json')[0]))['eval'].items()}
R = {}
for f in glob.glob('mis_mainC/mainC_*.json'):
    d = json.load(open(f)); c = d['cond']; R[(c['dose'], c['seed'])] = {m: np.array(v['correct']) for m, v in d['eval'].items()}
doses = sorted({k[0] for k in R}); seeds = sorted({k[1] for k in R})
def ch(m):
    t, f = m.rsplit('_', 1); return 0.5 if t == 'AGREE' else (0.0 if f == 'GEN' else 0.25)
def cv(m, x, s): return (R[(x, s)][m].mean() - ch(m)) / (base[m].mean() - ch(m))
def curve(m, ss): return np.mean([[cv(m, x, s) for x in doses] for s in ss], 0)
def d50(cs):
    cs = np.r_[1.0, cs]
    for i in range(1, len(cs)):
        if cs[i] < 0.5: return (i - 1) + (cs[i - 1] - 0.5) / (cs[i - 1] - cs[i]) if cs[i - 1] > cs[i] else float(i)
    return np.inf
def ranks(v):
    v = np.array([1e9 if (x is None or np.isinf(x)) else x for x in v]); o = v.argsort(); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def sp(a, b):
    ra, rb = ranks(a), ranks(b); return round(float(np.corrcoef(ra, rb)[0, 1]), 3) if ra.std() > 0 and rb.std() > 0 else None
D = {m: d50(curve(m, seeds)) for m in usable}
out = {'doses': doses, 'seeds': seeds, 'D50': {m: (None if np.isinf(v) else round(v, 2)) for m, v in D.items()}, 'curves': {m: [round(float(x), 3) for x in curve(m, seeds)] for m in usable}}
out['n_crossing'] = int(sum(not np.isinf(v) for v in D.values()))
sh = []
for a in itertools.combinations(seeds, 2):
    b = [s for s in seeds if s not in a]; sh.append(sp([d50(curve(m, a)) for m in usable], [d50(curve(m, b)) for m in usable]))
out['C1_split_half'] = sh; out['C1_median'] = float(np.median([x for x in sh if x is not None])); out['C1'] = out['C1_median'] >= 0.8
v = [D[m] for m in usable]
out['C2_noise_vs_quant'] = sp(v, [A3['D50']['quant'][m] for m in usable]); out['C2_noise_vs_ld_rand'] = sp(v, [A3['D50']['ld_rand'][m] for m in usable])
out['C2'] = bool((out['C2_noise_vs_quant'] or 0) >= 0.7 and (out['C2_noise_vs_ld_rand'] or 0) >= 0.7)
common = [m for m in usable if m in Q1['usable']]
out['C3_cross_model'] = sp([D[m] for m in common], [Q1['D50']['noise'][m] for m in common]); out['C3'] = bool((out['C3_cross_model'] or 0) >= 0.7)
c4 = {}
for f in ('GEN', 'CHOICE'):
    x = [D[f'ARITH{i}_{f}'] for i in (1, 2, 3)]; c4[f] = {'D50_1_2_3': [None if np.isinf(z) else round(z, 2) for z in x], 'ordered': bool(x[2] <= x[1] <= x[0]), 'informative': not all(np.isinf(z) for z in x)}
out['C4_cells'] = c4; out['C4'] = all(z['ordered'] for z in c4.values()) and any(z['informative'] for z in c4.values())
gc = [m for m in usable if not m.endswith('LETTER')]
out['seed_range_of_mean_c'] = {str(x): [round(float(min(np.mean([cv(m, x, s) for m in gc]) for s in seeds)), 2), round(float(max(np.mean([cv(m, x, s) for m in gc]) for s in seeds)), 2)] for x in doses}
out['other_within_mistral'] = {'quant-ld_rand': A3['within_model']['quant-ld_rand']}
out['all_three_cross_model'] = {'noise': out['C3_cross_model'], 'quant': A3['cross_model']['quant'], 'ld_rand': A3['cross_model']['ld_rand']}
json.dump(out, open('mis_mainC/analysis4.json', 'w'), indent=1, ensure_ascii=False)
for k in ['n_crossing', 'C1_split_half', 'C1_median', 'C1', 'C2_noise_vs_quant', 'C2_noise_vs_ld_rand', 'C2', 'C3_cross_model', 'C3', 'C4_cells', 'C4', 'seed_range_of_mean_c', 'all_three_cross_model']: print(k, out[k])
print('== Mistral noise', doses)
for m in sorted(usable, key=lambda z: 1e9 if np.isinf(D[z]) else D[z]): print(f"   {m:14s} D50={out['D50'][m]}  {out['curves'][m]}   | Qwen noise D50={Q1['D50']['noise'].get(m)}  | Mistral quant={A3['D50']['quant'][m]} ld_rand={A3['D50']['ld_rand'][m]}")
