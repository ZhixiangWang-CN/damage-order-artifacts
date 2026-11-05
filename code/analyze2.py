"""Locked analysis for PROTOCOL_ADDENDUM_1 (layer-drop position). Written before any main2 result was opened. python3 analyze2.py"""
import json, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def load(pat):
    R = {}
    for f in glob.glob(pat):
        d = json.load(open(f)); c = d['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = {m: np.array(v['correct']) for m, v in d['eval'].items()}
    return R
R1 = load('main/main_*.json'); R2 = load('main2/main2_*.json'); base = R1[('none', 0, 0)]
A1 = json.load(open('main/analysis.json')); usable = A1['usable']
def ch(m):
    t, f = m.rsplit('_', 1); return 0.5 if t == 'AGREE' else (0.0 if f == 'GEN' else 0.25)
def cv(R, m, k): return (R[k][m].mean() - ch(m)) / (base[m].mean() - ch(m))
NS = sorted({k[1] for k in R2 if k[0] == 'ld_early'}); KINDS = ['ld_early', 'ld_mid', 'ld_late', 'ld_rand']; seeds = sorted({k[2] for k in R2 if k[0] == 'ld_rand'})
def curve(kind, m, seed=None):
    if kind == 'ld_rand' and seed is None: return np.mean([[cv(R2, m, (kind, n, s)) for n in NS] for s in seeds], 0)
    return np.array([cv(R2, m, (kind, n, seed or 0)) for n in NS])
def d50(xs, cs):
    xs = np.r_[0, xs]; cs = np.r_[1.0, cs]
    for i in range(1, len(cs)):
        if cs[i] < 0.5: return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (cs[i - 1] - 0.5) / (cs[i - 1] - cs[i])) if cs[i - 1] > cs[i] else float(xs[i])
    return np.inf
def ranks(v):
    v = np.array([1e9 if np.isinf(x) else x for x in v]); o = v.argsort(); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def sp(a, b):
    ra, rb = ranks(a), ranks(b); return round(float(np.corrcoef(ra, rb)[0, 1]), 3) if ra.std() > 0 and rb.std() > 0 else None
D = {k: {m: d50(NS, curve(k, m)) for m in usable} for k in KINDS}
vec = lambda d: [d[m] for m in usable]
pil = {'noise': [np.inf if A1['D50']['noise'][m] is None else A1['D50']['noise'][m] for m in usable], 'quant': [np.inf if A1['D50']['quant'][m] is None else A1['D50']['quant'][m] for m in usable], 'pilot_layerdrop': [np.inf if A1['D50']['layerdrop'][m] is None else A1['D50']['layerdrop'][m] for m in usable]}
out = {'usable': usable, 'doses_n': NS, 'D50_layers': {k: {m: (None if np.isinf(v) else round(v, 2)) for m, v in D[k].items()} for k in KINDS}, 'curves': {k: {m: [round(float(x), 3) for x in curve(k, m)] for m in usable} for k in KINDS}}
late6 = {m: d50(NS[:6], curve('ld_late', m)[:6]) for m in usable}
out['A0_repro_spearman'] = sp(vec(late6), pil['pilot_layerdrop']); out['A0_max_abs_curve_diff'] = round(float(max(abs(np.array(A1['curves']['layerdrop'][m]) - curve('ld_late', m)[:6]).max() for m in usable)), 3); out['A0'] = (out['A0_repro_spearman'] or 0) >= 0.9
out['between_positions'] = {f'{a}-{b}': sp(vec(D[a]), vec(D[b])) for a, b in itertools.combinations(KINDS, 2)}
out['A1'] = bool((out['between_positions']['ld_mid-ld_late'] or 1) < 0.7 and (out['between_positions']['ld_early-ld_late'] or 1) < 0.7)
out['vs_noise'] = {k: sp(vec(D[k]), pil['noise']) for k in KINDS}; out['vs_quant'] = {k: sp(vec(D[k]), pil['quant']) for k in KINDS}
out['A2'] = bool((out['vs_noise']['ld_rand'] or 0) >= 0.7)
a3 = {}
for k in KINDS:
    for f in ('GEN', 'CHOICE'):
        v = [D[k][f'ARITH{i}_{f}'] for i in (1, 2, 3)]; a3[f'{k}/{f}'] = {'D50_1_2_3': [None if np.isinf(x) else round(x, 2) for x in v], 'holds': bool(v[2] <= v[1] <= v[0])}
out['A3_cells'] = a3; out['A3'] = sum(x['holds'] for x in a3.values()) >= 7
out['A4_FACT_GEN_D50'] = {k: out['D50_layers'][k]['FACT_GEN'] for k in KINDS}; out['A4'] = bool(D['ld_late']['FACT_GEN'] < D['ld_mid']['FACT_GEN'] and D['ld_late']['FACT_GEN'] < D['ld_early']['FACT_GEN'])
sd = {s: [d50(NS, curve('ld_rand', m, seed=s)) for m in usable] for s in seeds}
out['rand_seed_spearman'] = {f'{a}-{b}': sp(sd[a], sd[b]) for a, b in itertools.combinations(seeds, 2)}
out['mean_c_by_n'] = {k: [round(float(np.mean([curve(k, m)[i] for m in usable if not m.endswith('LETTER')])), 2) for i in range(len(NS))] for k in KINDS}
out['n_crossing'] = {k: int(sum(not np.isinf(D[k][m]) for m in usable)) for k in KINDS}
json.dump(out, open('main2/analysis2.json', 'w'), indent=1, ensure_ascii=False)
for k in ['A0_repro_spearman', 'A0_max_abs_curve_diff', 'A0', 'between_positions', 'A1', 'vs_noise', 'vs_quant', 'A2', 'A3_cells', 'A3', 'A4_FACT_GEN_D50', 'A4', 'rand_seed_spearman', 'mean_c_by_n', 'n_crossing']: print(k, out[k])
for k in KINDS:
    print('==', k)
    for m in sorted(usable, key=lambda z: 1e9 if np.isinf(D[k][z]) else D[k][z]): print(f"   {m:14s} D50={out['D50_layers'][k][m]}  {out['curves'][k][m]}")
fig, ax = plt.subplots(1, 5, figsize=(26, 6.5)); mo = sorted(usable, key=lambda z: np.mean([min(D[k][z], 11) for k in KINDS]))
tt = {'ld_early': 'early block', 'ld_mid': 'middle block', 'ld_late': 'late block', 'ld_rand': 'random layers (3 seeds)'}
for a, k in zip(ax, KINDS):
    M = np.array([np.r_[1.0, curve(k, m)] for m in mo]); a.imshow(np.clip(M, 0, 1), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    a.set_xticks(range(M.shape[1])); a.set_xticklabels(['0'] + [str(x) for x in NS]); a.set_yticks(range(len(mo))); a.set_yticklabels(mo, fontsize=8); a.set_title('Layers removed: ' + tt[k]); a.set_xlabel('layers removed')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='w' if M[i, j] < 0.6 else 'k')
a = ax[4]; cols = ['noise'] + KINDS; rk = {'noise': ranks([dict(zip(usable, pil['noise']))[m] for m in mo])}; rk.update({k: ranks([D[k][m] for m in mo]) for k in KINDS})
for i, m in enumerate(mo): a.plot(range(len(cols)), [rk[c][i] for c in cols], marker='o', label=m, ls='-' if 'GEN' in m else ('--' if 'CHOICE' in m else ':'))
a.set_xticks(range(len(cols))); a.set_xticklabels(['noise', 'early', 'mid', 'late', 'random'], fontsize=8); a.invert_yaxis(); a.set_ylabel('fragility rank (1 = fails first)'); a.grid(alpha=.3); a.legend(fontsize=6, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.45)); a.set_title('Fragility rank')
fig.suptitle('Addendum 1: where the layers are removed (Qwen2.5-7B-Instruct, chance-corrected, 200 items)', fontsize=11); fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig('main2/atlas_layerpos.png', dpi=120)
