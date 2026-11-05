"""Locked analysis for PROTOCOL_ADDENDUM_2 (second model). Written before any mainB result was opened. python3 analyze3.py"""
import json, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = {}
for f in glob.glob('mis_main/mainB_*.json'):
    d = json.load(open(f)); c = d['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = {m: np.array(v['correct']) for m, v in d['eval'].items()}
base = R[('none', 0, 0)]; MS = sorted(base)
def ch(m):
    t, f = m.rsplit('_', 1); return 0.5 if t == 'AGREE' else (0.0 if f == 'GEN' else 0.25)
usable = [m for m in MS if base[m].mean() - ch(m) >= 0.3]
dose = {'noise': sorted({k[1] for k in R if k[0] == 'noise'}), 'quant': sorted({k[1] for k in R if k[0] == 'quant'}, reverse=True)}
NS = sorted({k[1] for k in R if k[0] == 'ld_mid'}); LK = ['ld_rand', 'ld_mid', 'ld_late']
for k in LK: dose[k] = NS
seeds = {k: sorted({x[2] for x in R if x[0] == k}) for k in dose}
def cv(m, key): return (R[key][m].mean() - ch(m)) / (base[m].mean() - ch(m))
def curve(k, m, seed=None):
    if len(seeds[k]) > 1 and seed is None: return np.mean([[cv(m, (k, x, s)) for x in dose[k]] for s in seeds[k]], 0)
    return np.array([cv(m, (k, x, seed if seed is not None else seeds[k][0])) for x in dose[k]])
def d50(k, cs):
    xs = np.r_[0, dose[k]] if k in LK else np.arange(len(cs) + 1); cs = np.r_[1.0, cs]
    for i in range(1, len(cs)):
        if cs[i] < 0.5: return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (cs[i - 1] - 0.5) / (cs[i - 1] - cs[i])) if cs[i - 1] > cs[i] else float(xs[i])
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
K = list(dose); D = {k: {m: d50(k, curve(k, m)) for m in usable} for k in K}
vec = lambda k, ms=None: [D[k][m] for m in (ms or usable)]
out = {'baseline': {m: round(float(base[m].mean()), 3) for m in MS}, 'usable': usable, 'unusable': [m for m in MS if m not in usable], 'doses': dose,
       'D50': {k: {m: (None if np.isinf(v) else round(v, 2)) for m, v in D[k].items()} for k in K}, 'curves': {k: {m: [round(float(x), 3) for x in curve(k, m)] for m in usable} for k in K}}
sdn = {s: [d50('noise', curve('noise', m, seed=s)) for m in usable] for s in seeds['noise']}
out['B1_seed_spearman'] = {f'{a}-{b}': sp(sdn[a], sdn[b]) for a, b in itertools.combinations(seeds['noise'], 2)}; out['B1'] = bool(min(out['B1_seed_spearman'].values()) >= 0.8)
b2 = {}
for k in ('noise', 'quant', 'ld_rand'):
    for f in ('GEN', 'CHOICE'):
        ms = [f'ARITH{i}_{f}' for i in (1, 2, 3)]
        if all(m in usable for m in ms):
            v = [D[k][m] for m in ms]; b2[f'{k}/{f}'] = {'D50_1_2_3': [None if np.isinf(x) else round(x, 2) for x in v], 'holds': bool(v[2] <= v[1] <= v[0])}
out['B2_cells'] = b2; out['B2_fraction'] = round(sum(x['holds'] for x in b2.values()) / max(1, len(b2)), 2); out['B2'] = out['B2_fraction'] >= 5 / 6 - 1e-9
b3 = {}
for k in K:
    for t in ['ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'AGREE']:
        g, c = f'{t}_GEN', f'{t}_CHOICE'
        if g in usable and c in usable: b3[f'{k}/{t}'] = bool(D[k][g] <= D[k][c])
out['B3_fraction'] = round(sum(b3.values()) / max(1, len(b3)), 2); out['B3_fails'] = [k for k, v in b3.items() if not v]; out['B3'] = out['B3_fraction'] >= 0.75
out['within_model'] = {f'{a}-{b}': sp(vec(a), vec(b)) for a, b in itertools.combinations(K, 2)}
out['B4'] = bool(min(out['within_model'][x] or 0 for x in ('noise-quant', 'noise-ld_rand', 'quant-ld_rand')) >= 0.7)
out['B5'] = bool((out['within_model']['ld_mid-ld_late'] or 1) < 0.7)
Q1 = json.load(open('main/analysis.json')); Q2 = json.load(open('main2/analysis2.json'))
common = [m for m in usable if m in Q1['usable']]; out['common_measures'] = common
qn = [Q1['D50']['noise'][m] for m in common]; qq = [Q1['D50']['quant'][m] for m in common]; qr = [Q2['D50_layers']['ld_rand'][m] for m in common]
out['cross_model'] = {'noise': sp(vec('noise', common), qn), 'quant': sp(vec('quant', common), qq), 'ld_rand': sp(vec('ld_rand', common), qr), 'ld_mid': sp(vec('ld_mid', common), [Q2['D50_layers']['ld_mid'][m] for m in common]), 'ld_late': sp(vec('ld_late', common), [Q2['D50_layers']['ld_late'][m] for m in common])}
out['B6'] = bool((out['cross_model']['noise'] or 0) >= 0.7)
out['B7_detail'] = {'RETR_GEN': {k: out['D50'][k].get('RETR_GEN') for k in ('ld_mid', 'ld_late')}, 'FACT_GEN': {k: out['D50'][k].get('FACT_GEN') for k in ('ld_mid', 'ld_late')}}
out['B7'] = bool('RETR_GEN' in usable and 'FACT_GEN' in usable and D['ld_mid']['RETR_GEN'] < D['ld_late']['RETR_GEN'] and D['ld_late']['FACT_GEN'] < D['ld_mid']['FACT_GEN'])
sdr = {s: [d50('ld_rand', curve('ld_rand', m, seed=s)) for m in usable] for s in seeds['ld_rand']}
out['rand_seed_spearman'] = {f'{a}-{b}': sp(sdr[a], sdr[b]) for a, b in itertools.combinations(seeds['ld_rand'], 2)}
out['n_crossing'] = {k: int(sum(not np.isinf(D[k][m]) for m in usable)) for k in K}
json.dump(out, open('mis_main/analysis3.json', 'w'), indent=1, ensure_ascii=False)
for k in ['baseline', 'unusable', 'B1_seed_spearman', 'B1', 'B2_cells', 'B2', 'B3_fraction', 'B3_fails', 'B3', 'within_model', 'B4', 'B5', 'common_measures', 'cross_model', 'B6', 'B7_detail', 'B7', 'rand_seed_spearman', 'n_crossing']: print(k, out[k])
for k in K:
    print('==', k, dose[k])
    for m in sorted(usable, key=lambda z: 1e9 if np.isinf(D[k][z]) else D[k][z]): print(f"   {m:14s} D50={out['D50'][k][m]}  {out['curves'][k][m]}")
fig, ax = plt.subplots(1, 6, figsize=(31, 6.5)); mo = sorted(usable, key=lambda z: np.mean([ranks(vec(k))[usable.index(z)] for k in ('noise', 'quant', 'ld_rand')]))
tt = {'noise': 'Gaussian noise (r)', 'quant': 'Quantisation (q)', 'ld_rand': 'Random layers removed', 'ld_mid': 'Middle block removed', 'ld_late': 'Late block removed'}
for a, k in zip(ax, K):
    M = np.array([np.r_[1.0, curve(k, m)] for m in mo]); a.imshow(np.clip(M, 0, 1), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    a.set_xticks(range(M.shape[1])); a.set_xticklabels(['0'] + [str(x) for x in dose[k]], fontsize=7); a.set_yticks(range(len(mo))); a.set_yticklabels(mo, fontsize=8); a.set_title(tt[k])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='w' if M[i, j] < 0.6 else 'k')
a = ax[5]
for i, m in enumerate(mo): a.plot(range(len(K)), [ranks(vec(k, mo))[i] for k in K], marker='o', label=m, ls='-' if 'GEN' in m else ('--' if 'CHOICE' in m else ':'))
a.set_xticks(range(len(K))); a.set_xticklabels(['noise', 'quant', 'random', 'mid', 'late'], fontsize=8); a.invert_yaxis(); a.set_ylabel('fragility rank (1 = fails first)'); a.grid(alpha=.3); a.legend(fontsize=6, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.45)); a.set_title('Fragility rank')
fig.suptitle('Addendum 2: Mistral-7B-Instruct-v0.3, chance-corrected scores (200 items; noise and random layers = mean of 3 seeds)', fontsize=11); fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig('mis_main/atlas_mistral.png', dpi=110)
