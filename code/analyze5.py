"""Locked analysis for PROTOCOL_ADDENDUM_4 (standard benchmarks). Written before any bench result was opened. python3 analyze5.py"""
import json, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = {}; CH = {}
for f in glob.glob('bench/bench_*.json'):
    d = json.load(open(f)); c = d['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = {m: np.array(v['correct']) for m, v in d['eval'].items()}
    for m, v in d['eval'].items(): CH[m] = v['chance']
base = R[('none', 0, 0)]; MS = sorted(base); OLD = ('ARITH', 'FACT', 'RETR')
usable = [m for m in MS if base[m].mean() - CH[m] >= 0.3]; bench = [m for m in usable if not m.startswith(OLD)]; old = [m for m in usable if m.startswith(OLD)]
dose = {'noise': sorted({k[1] for k in R if k[0] == 'noise'}), 'quant': sorted({k[1] for k in R if k[0] == 'quant'}, reverse=True), 'ld_rand': sorted({k[1] for k in R if k[0] == 'ld_rand'})}
seeds = {k: sorted({x[2] for x in R if x[0] == k}) for k in dose}; K = list(dose)
def cv(m, key): return (R[key][m].mean() - CH[m]) / (base[m].mean() - CH[m])
def curve(k, m, ss=None): return np.mean([[cv(m, (k, x, s)) for x in dose[k]] for s in (ss or seeds[k])], 0)
def d50(k, cs):
    xs = np.r_[0, dose[k]] if k == 'ld_rand' else np.arange(len(cs) + 1); cs = np.r_[1.0, cs]
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
D = {k: {m: d50(k, curve(k, m)) for m in usable} for k in K}; vec = lambda k, ms: [D[k][m] for m in ms]
out = {'baseline': {m: round(float(base[m].mean()), 3) for m in MS}, 'chance': CH, 'usable_bench': bench, 'usable_old': old, 'unusable': [m for m in MS if m not in usable], 'doses': dose,
       'D50': {k: {m: (None if np.isinf(v) else round(v, 2)) for m, v in D[k].items()} for k in K}, 'curves': {k: {m: [round(float(x), 3) for x in curve(k, m)] for m in usable} for k in K}}
out['n_crossing_bench'] = {k: int(sum(not np.isinf(D[k][m]) for m in bench)) for k in K}
sh = []
for a in itertools.combinations(seeds['noise'], 2):
    b = [s for s in seeds['noise'] if s not in a]; sh.append(sp([d50('noise', curve('noise', m, a)) for m in bench], [d50('noise', curve('noise', m, b)) for m in bench]))
out['D1_split_half'] = sh; out['D1_median'] = float(np.median([x for x in sh if x is not None])) if any(x is not None for x in sh) else None; out['D1'] = bool((out['D1_median'] or 0) >= 0.8)
pairs = list(itertools.combinations(K, 2))
out['D2_bench'] = {f'{a}-{b}': sp(vec(a, bench), vec(b, bench)) for a, b in pairs}; out['D2'] = bool(min((v or 0) for v in out['D2_bench'].values()) >= 0.7)
d3 = {}
for k in K:
    for t in sorted({m.rsplit('_', 1)[0] for m in bench}):
        g = next((f'{t}_{f}' for f in ('GEN', 'GENOPT') if f'{t}_{f}' in bench), None); c = f'{t}_CHOICE'
        if g and c in bench: d3[f'{k}/{t}'] = bool(D[k][g] <= D[k][c])
out['D3_cells'] = d3; out['D3_fraction'] = round(sum(d3.values()) / max(1, len(d3)), 2); out['D3'] = out['D3_fraction'] >= 0.75
d4 = {}
for k in K:
    for g in ('GSM8K_COT', 'GSM8K_CHOICE', 'GSM8K_GEN'):
        if g in bench and 'SCIQ_CHOICE' in bench: d4[f'{k}/{g}<SCIQ_CHOICE'] = bool(D[k][g] < D[k]['SCIQ_CHOICE'])
out['D4_cells'] = d4; core = [v for kk, v in d4.items() if 'GSM8K_GEN' not in kk]; out['D4'] = sum(core) >= 5 and len(core) == 6
P = json.load(open('main/analysis.json')); co = [m for m in old if m in P['usable']]
out['D5_repro'] = sp(vec('noise', co), [P['D50']['noise'][m] for m in co]); out['D5'] = bool((out['D5_repro'] or 0) >= 0.9)
out['D6_all'] = {f'{a}-{b}': sp(vec(a, usable), vec(b, usable)) for a, b in pairs}; out['D6'] = bool(min((v or 0) for v in out['D6_all'].values()) >= 0.7)
out['old_only'] = {f'{a}-{b}': sp(vec(a, old), vec(b, old)) for a, b in pairs}
out['explore'] = {'GSM8K': {k: {f: out['D50'][k].get(f'GSM8K_{f}') for f in ('GEN', 'COT', 'CHOICE')} for k in K}, 'MMLU': {k: {m: out['D50'][k].get(m) for m in usable if m.startswith('MMLU')} for k in K}}
ok8 = len(bench) >= 8 and min(out['n_crossing_bench'].values()) >= 6; out['measurable'] = bool(ok8)
json.dump(out, open('bench/analysis5.json', 'w'), indent=1, ensure_ascii=False)
for k in ['baseline', 'unusable', 'n_crossing_bench', 'measurable', 'D1_split_half', 'D1_median', 'D1', 'D2_bench', 'D2', 'D3_cells', 'D3_fraction', 'D3', 'D4_cells', 'D4', 'D5_repro', 'D5', 'D6_all', 'D6', 'old_only', 'explore']: print(k, out[k])
for k in K:
    print('==', k, dose[k])
    for m in sorted(usable, key=lambda z: 1e9 if np.isinf(D[k][z]) else D[k][z]): print(f"   {m:18s} D50={out['D50'][k][m]}  {out['curves'][k][m]}")
fig, ax = plt.subplots(1, 4, figsize=(24, 9), gridspec_kw={'width_ratios': [1, 1, 1.2, 1.1]}); mo = sorted(usable, key=lambda z: np.mean([ranks(vec(k, usable))[usable.index(z)] for k in K]))
tt = {'noise': 'Gaussian weight noise (r), 5 seeds', 'quant': 'Quantisation (levels q)', 'ld_rand': 'Random layers removed (n), 3 seeds'}
for a, k in zip(ax, K):
    M = np.array([np.r_[1.0, curve(k, m)] for m in mo]); a.imshow(np.clip(M, 0, 1), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    a.set_xticks(range(M.shape[1])); a.set_xticklabels(['0'] + [str(x) for x in dose[k]], fontsize=7); a.set_yticks(range(len(mo))); a.set_yticklabels([('* ' if m in bench else '') + m for m in mo], fontsize=7); a.set_title(tt[k], fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5, color='w' if M[i, j] < 0.6 else 'k')
a = ax[3]
for i, m in enumerate(mo): a.plot(range(3), [ranks(vec(k, mo))[i] for k in K], marker='o', lw=1.6 if m in bench else .7, alpha=1 if m in bench else .45, label=m if m in bench else None)
a.set_xticks(range(3)); a.set_xticklabels(['noise', 'quant', 'random layers']); a.invert_yaxis(); a.set_ylabel('fragility rank (1 = fails first)'); a.grid(alpha=.3); a.legend(fontsize=6, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.3)); a.set_title('Fragility rank (bold = benchmark measures)')
fig.suptitle('Addendum 4: standard benchmarks, Qwen2.5-7B-Instruct, chance-corrected (* = benchmark measure)', fontsize=11); fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig('bench/atlas_bench.png', dpi=110)
