"""Locked analysis for the damage-atlas pilot (PROTOCOL.md). Written before the main results were opened. python3 analyze.py main"""
import json, glob, sys, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1]; rng = np.random.default_rng(0)
R = {}
for f in glob.glob(f'{D}/main_*.json'):
    d = json.load(open(f)); c = d['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = {m: np.array(v['correct']) for m, v in d['eval'].items()}
base = R[('none', 0, 0)]; MS = sorted(base)
def ch(m):
    t, f = m.rsplit('_', 1); return 0.5 if t == 'AGREE' else (0.0 if f == 'GEN' else 0.25)
usable = [m for m in MS if base[m].mean() - ch(m) >= 0.3]
order = {'noise': sorted({k[1] for k in R if k[0] == 'noise'}), 'quant': sorted({k[1] for k in R if k[0] == 'quant'}, reverse=True), 'layerdrop': sorted({k[1] for k in R if k[0] == 'layerdrop'})}
seeds = sorted({k[2] for k in R if k[0] == 'noise'})
def cval(m, dmg, dose, seed, idx=None):
    a = R[(dmg, dose, seed)][m]; b = base[m]
    if idx is not None: a = a[idx]; b = b[idx]
    den = b.mean() - ch(m)
    return (a.mean() - ch(m)) / den if den > 1e-9 else np.nan
def curve(m, dmg, seed=None, idx=None):
    if dmg == 'noise' and seed is None: return np.mean([[cval(m, dmg, x, s, idx) for x in order[dmg]] for s in seeds], 0)
    return np.array([cval(m, dmg, x, seed or 0, idx) for x in order[dmg]])
def d50(cs):
    cs = np.r_[1.0, cs]
    for i in range(1, len(cs)):
        if cs[i] < 0.5:
            return (i - 1) + (cs[i - 1] - 0.5) / (cs[i - 1] - cs[i]) if cs[i - 1] > cs[i] else float(i)
    return np.inf
def ranks(v):
    v = np.array([1e9 if np.isinf(x) else x for x in v]); o = v.argsort(); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')
out = {'baseline': {m: float(base[m].mean()) for m in MS}, 'usable': usable, 'unusable': [m for m in MS if m not in usable], 'doses': order}
D50 = {dmg: {m: d50(curve(m, dmg)) for m in usable} for dmg in order}
out['D50'] = {dmg: {m: (None if np.isinf(v) else round(float(v), 2)) for m, v in D50[dmg].items()} for dmg in order}
out['curves'] = {dmg: {m: [round(float(x), 3) for x in curve(m, dmg)] for m in usable} for dmg in order}
# bootstrap CIs
n = len(base[MS[0]]); boot = {dmg: {m: [] for m in usable} for dmg in order}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    for dmg in order:
        for m in usable: boot[dmg][m].append(d50(curve(m, dmg, idx=idx)))
def ci(v):
    v = np.array([99 if np.isinf(x) or np.isnan(x) else x for x in v]); lo, hi = np.percentile(v, [2.5, 97.5]); return [None if lo > 90 else round(float(lo), 2), None if hi > 90 else round(float(hi), 2)]
out['D50_ci95'] = {dmg: {m: ci(boot[dmg][m]) for m in usable} for dmg in order}
# P1
sd = {s: [d50(curve(m, 'noise', seed=s)) for m in usable] for s in seeds}
out['P1_seed_spearman'] = {f'{a}-{b}': round(spearman(sd[a], sd[b]), 3) for a, b in itertools.combinations(seeds, 2)}
out['P1'] = bool(min(out['P1_seed_spearman'].values()) >= 0.8)
# P2
p2 = {}
for dmg in order:
    for f in ('GEN', 'CHOICE'):
        ms = [f'ARITH{k}_{f}' for k in (1, 2, 3)]
        if all(m in usable for m in ms):
            v = [D50[dmg][m] for m in ms]; p2[f'{dmg}/{f}'] = {'D50_1_2_3': [None if np.isinf(x) else round(float(x), 2) for x in v], 'holds': bool(v[2] <= v[1] <= v[0])}
out['P2_cells'] = p2; out['P2'] = sum(x['holds'] for x in p2.values()) >= 5
# P3
p3 = {}
for dmg in order:
    for t in ['ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'AGREE']:
        g, c = f'{t}_GEN', f'{t}_CHOICE'
        if g in usable and c in usable: p3[f'{dmg}/{t}'] = {'GEN': out['D50'][dmg][g], 'CHOICE': out['D50'][dmg][c], 'holds': bool(D50[dmg][g] <= D50[dmg][c])}
out['P3_cells'] = p3; out['P3_fraction'] = round(sum(x['holds'] for x in p3.values()) / max(1, len(p3)), 2); out['P3'] = out['P3_fraction'] >= 0.75
# P4
out['P4_cross_spearman'] = {f'{a}-{b}': round(spearman([D50[a][m] for m in usable], [D50[b][m] for m in usable]), 3) for a, b in itertools.combinations(order, 2)}
out['P4'] = bool(min(out['P4_cross_spearman'].values()) >= 0.7)
# go / stop
out['n_usable_crossing'] = {dmg: int(sum(not np.isinf(D50[dmg][m]) for m in usable)) for dmg in order}
minseed = min(out['P1_seed_spearman'].values()); mincross = min(out['n_usable_crossing'].values())
out['decision'] = 'STOP' if (minseed < 0.5 or mincross < 6) else ('GO' if (out['P1'] and mincross >= 8) else 'IN-BETWEEN')
# exploratory: noise-equivalent r
nx = np.r_[0.0, order['noise']]; eq = {}
for dmg in ('quant', 'layerdrop'):
    eq[dmg] = {}
    for j, x in enumerate(order[dmg]):
        row = {}
        for m in usable:
            ny = np.minimum.accumulate(np.r_[1.0, curve(m, 'noise')]); v = curve(m, dmg)[j]
            row[m] = None if (v >= ny[0] - 0.02 or v < ny[-1]) else round(float(np.interp(-v, -ny, nx)), 3)
        vals = [z for z in row.values() if z is not None]
        eq[dmg][str(x)] = {'per_measure': row, 'n': len(vals), 'min': min(vals) if vals else None, 'max': max(vals) if vals else None}
out['noise_equivalent_r'] = eq
json.dump(out, open(f'{D}/analysis.json', 'w'), indent=1, ensure_ascii=False)
print('usable', len(usable), '| unusable', out['unusable'])
print('baseline', {m: round(v, 2) for m, v in out['baseline'].items()})
for dmg in order:
    print('==', dmg, order[dmg])
    for m in sorted(usable, key=lambda z: (1e9 if np.isinf(D50[dmg][z]) else D50[dmg][z])):
        print(f"   {m:14s} D50={out['D50'][dmg][m]}  ci={out['D50_ci95'][dmg][m]}  curve={out['curves'][dmg][m]}")
for k in ['P1_seed_spearman', 'P1', 'P2_cells', 'P2', 'P3_fraction', 'P3', 'P4_cross_spearman', 'P4', 'n_usable_crossing', 'decision']: print(k, out[k])
print('P3 fails:', [k for k, v in p3.items() if not v['holds']])
for dmg in eq:
    for x, v in eq[dmg].items(): print('equiv r', dmg, x, 'n', v['n'], 'range', v['min'], v['max'])
# figure
fig, ax = plt.subplots(1, 4, figsize=(21, 6.5), gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
lab = {'noise': 'Gaussian weight noise (r)', 'quant': 'Quantisation (levels q)', 'layerdrop': 'Layers removed (n)'}
mo = sorted(usable, key=lambda z: np.mean([min(D50[d][z], 7) for d in order]))
for a, dmg in zip(ax, order):
    M = np.array([np.r_[1.0, curve(m, dmg)] for m in mo]); im = a.imshow(np.clip(M, 0, 1), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    a.set_xticks(range(M.shape[1])); a.set_xticklabels(['0'] + [str(x) for x in order[dmg]]); a.set_yticks(range(len(mo))); a.set_yticklabels(mo, fontsize=8); a.set_title(lab[dmg]); a.set_xlabel('dose')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='w' if M[i, j] < 0.6 else 'k')
a = ax[3]; rk = {dmg: ranks([D50[dmg][m] for m in mo]) for dmg in order}
for i, m in enumerate(mo): a.plot(range(3), [rk[d][i] for d in order], 'o-', label=m, lw=1.5 if 'GEN' in m else 1, ls='-' if 'GEN' in m else ('--' if 'CHOICE' in m else ':'))
a.set_xticks(range(3)); a.set_xticklabels(list(order)); a.set_ylabel('fragility rank (1 = fails first)'); a.invert_yaxis(); a.legend(fontsize=6, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.42)); a.set_title('Fragility rank by damage type'); a.grid(alpha=.3)
fig.suptitle('Damage-atlas pilot: Qwen2.5-7B-Instruct, chance-corrected scores (200 items; noise = mean of 3 seeds)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(f'{D}/atlas_pilot.png', dpi=130)
