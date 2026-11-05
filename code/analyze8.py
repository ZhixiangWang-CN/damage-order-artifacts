"""Locked analysis for PROTOCOL_ADDENDUM_7 (stability of the fragility exponent). Written before any result of this batch was opened. python3 analyze8.py"""
import json, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
MODELS = {'Qwen-1.5B': ['ext_q15'], 'Qwen-7B': ['ext_q7', 'fmt_qwen'], 'Qwen-14B': ['ext_q14'], 'Llama-3B': ['ext_llama'], 'Mistral-7B': ['ext_mis', 'fmt_mis'], 'Gemma-9B': ['ext_gemma'], 'Phi-3.5': ['ext_phi']}
UNT = ['noise', 'zero', 'neuron', 'ld_rand']; ALLD = UNT + ['ld_late']
def rate(e, m):
    if m in e['fmt']:
        v = e['fmt'][m]; return float(np.mean(np.array(v['format']) * np.array(v['content']))), 0.0
    v = e['ref'][m]; return float(np.mean(v['correct'])), v['chance']
def rank1(L, M, it=500):
    n = np.ones(L.shape[0]); h = np.ones(L.shape[1])
    for _ in range(it):
        h = np.array([(M[:, j] * n * L[:, j]).sum() / max((M[:, j] * n * n).sum(), 1e-12) for j in range(L.shape[1])])
        n = np.array([(M[i] * h * L[i]).sum() / max((M[i] * h * h).sum(), 1e-12) for i in range(L.shape[0])])
    s = np.exp(np.mean(np.log(n[n > 0]))); return n / s, h * s
def pear(a, b): return float(np.corrcoef(a, b)[0, 1])
N = {}; info = {}
for name, dirs in MODELS.items():
    N[name] = {}; info[name] = {}
    for d in dirs:
        R = {}
        for f in glob.glob(f'{d}/ext_*.json') + glob.glob(f'{d}/fmt_*.json'):
            x = json.load(open(f)); c = x['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = x
        if ('none', 0, 0) not in R: continue
        b = R[('none', 0, 0)]; ms = []
        for m in list(b['fmt']) + list(b['ref']):
            r0, ch = rate(b, m)
            if (r0 >= 0.8) if ch == 0 else (r0 - ch >= 0.3): ms.append(m)
        for k in ALLD:
            if k in N[name]: continue
            doses = sorted({x[1] for x in R if x[0] == k})
            if len(doses) < 3: continue
            seeds = sorted({x[2] for x in R if x[0] == k})
            C = np.array([[np.mean([(rate(R[(k, x, s)], m)[0] - rate(b, m)[1]) / (rate(b, m)[0] - rate(b, m)[1]) for s in seeds if (k, x, s) in R]) for x in doses] for m in ms])
            M = ((C > 0.03) & (C < 0.97)).astype(float); keep = M.sum(1) >= 2
            if keep.sum() < 12: info[name][k] = {'status': 'not measurable', 'n_fitted': int(keep.sum()), 'doses': doses}; continue
            n, h = rank1(-np.log(np.clip(C[keep], 1e-6, 1)), M[keep]); P = np.exp(-np.outer(n, h)); mm = M[keep] > 0; a, p = C[keep][mm], P[mm]
            N[name][k] = {m: float(v) for m, v in zip([m for m, q in zip(ms, keep) if q], n)}
            info[name][k] = {'status': 'ok', 'n_fitted': int(keep.sum()), 'doses': doses, 'n_seeds': len(seeds), 'R2_rank1': round(float(1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()), 3), 'usable_measures': len(ms)}
def corr(a, b):
    ms = [m for m in a if m in b]
    return (round(pear(np.log([a[m] for m in ms]), np.log([b[m] for m in ms])), 3), len(ms)) if len(ms) >= 10 else (None, len(ms))
out = {'info': info, 'n': {m: {k: {x: round(v, 3) for x, v in d.items()} for k, d in N[m].items()} for m in N}}
h1 = {}; h1all = []
for m in N:
    prs = {f'{a}-{b}': corr(N[m][a], N[m][b])[0] for a, b in itertools.combinations([k for k in UNT if k in N[m]], 2)}; v = [x for x in prs.values() if x is not None]
    h1[m] = {'pairs': prs, 'median': round(float(np.median(v)), 3) if v else None}; h1all += v
out['H1'] = {'by_model': h1, 'overall_median': round(float(np.median(h1all)), 3) if h1all else None, 'n_pairs': len(h1all), 'models_with_median_ge_0.6': int(sum((x['median'] or 0) >= 0.6 for x in h1.values())), 'n_models': len(h1)}
out['H1']['pass'] = bool((out['H1']['overall_median'] or 0) >= 0.7 and out['H1']['models_with_median_ge_0.6'] >= 6)
h2 = {}
for k in UNT:
    prs = {f'{a}|{b}': corr(N[a][k], N[b][k])[0] for a, b in itertools.combinations([m for m in N if k in N[m]], 2)}; v = [x for x in prs.values() if x is not None]
    h2[k] = {'median': round(float(np.median(v)), 3) if v else None, 'min': round(min(v), 3) if v else None, 'max': round(max(v), 3) if v else None, 'n_pairs': len(v), 'pairs': prs}
out['H2'] = h2; out['H2_pass'] = all((h2[k]['median'] or 0) >= 0.6 for k in UNT)
h3 = {}
for m in N:
    if 'noise' in N[m] and 'ld_late' in N[m] and h1[m]['median'] is not None:
        c = corr(N[m]['noise'], N[m]['ld_late'])[0]; h3[m] = {'noise_vs_ld_late': c, 'untargeted_median': h1[m]['median'], 'holds': bool(c is not None and c < h1[m]['median'])}
out['H3'] = h3; out['H3_pass'] = sum(v['holds'] for v in h3.values()) >= 5
h4 = {}
for m in N:
    for k in UNT:
        if k in N[m] and all(f'ARITH{i}_GEN' in N[m][k] for i in (1, 2, 3)):
            v = [N[m][k][f'ARITH{i}_GEN'] for i in (1, 2, 3)]; h4[f'{m}/{k}'] = {'n_1_2_3': [round(x, 2) for x in v], 'holds': bool(v[2] >= v[1] >= v[0])}
out['H4'] = h4; out['H4_fraction'] = round(sum(v['holds'] for v in h4.values()) / max(1, len(h4)), 3); out['H4_pass'] = bool(h4 and out['H4_fraction'] >= 0.8)
json.dump(out, open('ext_analysis8.json', 'w'), indent=1, ensure_ascii=False)
print('INFO'); [print('  ', m, {k: (v['status'], v['n_fitted'], v.get('R2_rank1'), v['doses']) for k, v in info[m].items()}) for m in info]
print('H1 pass', out['H1']['pass'], '| overall median', out['H1']['overall_median'], 'over', out['H1']['n_pairs'], 'pairs | models with median>=0.6:', out['H1']['models_with_median_ge_0.6'], '/', out['H1']['n_models'])
[print('  ', m, v['median'], v['pairs']) for m, v in h1.items()]
print('H2 pass', out['H2_pass']); [print('  ', k, 'median', v['median'], 'min', v['min'], 'max', v['max'], 'pairs', v['n_pairs']) for k, v in h2.items()]
print('H3 pass', out['H3_pass']); [print('  ', m, v) for m, v in h3.items()]
print('H4 pass', out['H4_pass'], out['H4_fraction'], 'of', len(h4)); print('   fails:', {k: v['n_1_2_3'] for k, v in h4.items() if not v['holds']})
ms = sorted({x for m in N for k in N[m] for x in N[m][k]}); cols = [(m, k) for m in N for k in ALLD if k in N[m]]
A = np.array([[np.log(N[m][k][x]) if x in N[m][k] else np.nan for (m, k) in cols] for x in ms]); o = np.argsort(np.nanmean(A, 1))
fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.55), 9)); im = ax.imshow(A[o], aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{m}\n{k}' for m, k in cols], fontsize=6, rotation=90); ax.set_yticks(range(len(ms))); ax.set_yticklabels([ms[i] for i in o], fontsize=7)
plt.colorbar(im, label='log fragility exponent (red = fails early)'); ax.set_title('Fragility exponent per measure, by model and damage type (white = not fitted)'); fig.tight_layout(); fig.savefig('ext_exponents.png', dpi=120)
