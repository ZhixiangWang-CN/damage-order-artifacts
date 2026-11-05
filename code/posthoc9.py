"""Post-hoc checks on addendum 8 (not preregistered).  python3 posthoc9.py runs_v8/server_a runs_v8/server_b
A. is the task effect just intact difficulty?   B. do different damages share one h-vs-KL curve?"""
import sys, os, json, gzip, numpy as np
from analyze9 import load_model, fit_cell, additive, MODELS, UNT, TASKS, FORMATS
roots = sys.argv[1:]; out = {'A': {}, 'B': {}}
def sp(a, b): ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
allA = []
for m in MODELS:
    dirs = [f'{r}/{m}' for r in roots if os.path.isdir(f'{r}/{m}')]; R, use = load_model(dirs); base = R[('none', 0, 0)]; ms = sorted(use['usable'])
    with gzip.open(f'{dirs[0]}/none_0_0.json.gz', 'rt') as fh: raw = json.load(fh)['res']
    cells = {}
    for k in UNT:
        r = fit_cell(R, base, ms, k)
        if r: cells[k] = (r, additive(r[0]))
    A = [c[1]['a'] for c in cells.values() if c[1]]
    tasks = [t for t in TASKS if sum(t in a for a in A) >= 2]
    eff = {t: np.mean([a[t] for a in A if t in a]) for t in tasks}
    # difficulty 1: intact chance-corrected accuracy averaged over usable formats; difficulty 2: intact forced-choice margin (gold minus best distractor)
    d1 = {t: np.mean([(raw[f'{t}_{f}']['acc'] - raw[f'{t}_{f}']['chance']) / (1 - raw[f'{t}_{f}']['chance']) for f in FORMATS if f'{t}_{f}' in use['usable']]) for t in tasks}
    d2 = {t: float(np.mean([s[0] - max(s[1:]) for s in raw[f'{t}_CHOICE']['scores']])) for t in tasks}
    a1 = sp([eff[t] for t in tasks], [-d1[t] for t in tasks]); a2 = sp([eff[t] for t in tasks], [-d2[t] for t in tasks])
    out['A'][m] = {'n_tasks': len(tasks), 'rho_effect_vs_inaccuracy': round(a1, 3), 'rho_effect_vs_neg_margin': round(a2, 3)}
    allA.append((m, eff, d1))
    # B: h vs KL pooled across the four damages; compare per-damage fits with one shared curve (log h = a + b log KL)
    pts = {k: [(c[0][2]['kl'][x], c[0][1][x]) for x in c[0][2]['doses'] if c[0][1][x] > 0 and c[0][2]['kl'][x] > 0] for k, c in cells.items()}
    X = np.array([np.log(p[0]) for v in pts.values() for p in v]); Y = np.array([np.log(p[1]) for v in pts.values() for p in v])
    def rss(x, y): A_ = np.c_[np.ones_like(x), x]; c = np.linalg.lstsq(A_, y, rcond=None)[0]; return float(((y - A_ @ c) ** 2).sum())
    pooled = rss(X, Y); sep = sum(rss(np.log([p[0] for p in v]), np.log([p[1] for p in v])) for v in pts.values() if len(v) >= 3); tot = float(((Y - Y.mean()) ** 2).sum())
    out['B'][m] = {'R2_one_shared_curve': round(1 - pooled / tot, 3), 'R2_separate_curves': round(1 - sep / tot, 3), 'n_points': int(len(X))}
# A across models: does task difficulty explain the cross-model agreement? partial out difficulty and re-correlate task effects between models
res = []
for (m1, e1, d1), (m2, e2, d2) in [(a, b) for i, a in enumerate(allA) for b in allA[i + 1:]]:
    ts = [t for t in e1 if t in e2]
    if len(ts) < 8: continue
    def resid(e, d): y = np.array([e[t] for t in ts]); x = np.array([d[t] for t in ts]); A_ = np.c_[np.ones_like(x), x]; return y - A_ @ np.linalg.lstsq(A_, y, rcond=None)[0]
    res.append((np.corrcoef([e1[t] for t in ts], [e2[t] for t in ts])[0, 1], np.corrcoef(resid(e1, d1), resid(e2, d2))[0, 1]))
out['A_cross_model'] = {'median_raw': round(float(np.median([r[0] for r in res])), 3), 'median_after_removing_intact_accuracy': round(float(np.median([r[1] for r in res])), 3), 'n_pairs': len(res)}
json.dump(out, open('posthoc9.json', 'w'), indent=1); print(json.dumps(out, indent=1))
