"""Locked analysis for PROTOCOL_ADDENDUM_9. Written with the protocol, before any compression or 32B condition existed.
  python3 analyze10.py <v9_root> <v8_root> [<v8_root2> ...]   -> analysis10.json + printed verdicts
Exponents, task and format effects come from analyze9.py (imported unchanged): same window, same rank-one fit, same additive model.
"""
import sys, os, json, itertools
import numpy as np
import analyze9 as Z

MODELS7 = ['q15', 'q7', 'q14', 'llama', 'mis', 'gemma', 'phi']; COMP = ['wanda', 'sparsegpt', 'gptq']; UNT = Z.UNT
EXTREME = ['COPY', 'RETR', 'FACT', 'ARITH3']; NPERM = 1000

def cells_for(dirs, damages):
    R, use = Z.load_model(dirs)
    if ('none', 0, 0) not in R or use is None: return None
    base = R[('none', 0, 0)]; ms = sorted(use['usable']); E, A, INFO = {}, {}, {}
    for k in damages:
        r = Z.fit_cell(R, base, ms, k)
        if r is None: INFO[k] = 'not measurable'; continue
        E[k], _, inf = r; INFO[k] = {kk: vv for kk, vv in inf.items() if kk != 'C'}; A[k] = Z.additive(E[k])
    return E, A, INFO

def reference(A, m):
    """mean of the untargeted task effects of one model, over tasks with at least two untargeted damages."""
    tes = [A[m][k]['a'] for k in UNT if A[m].get(k)]; ts = {t for te in tes for t in te}
    return {t: float(np.mean([te[t] for te in tes if t in te])) for t in ts if sum(t in te for te in tes) >= 2}

def main(v9, v8roots):
    E, A, INFO = {}, {}, {}
    for m in MODELS7 + ['q32']:
        dirs = [f'{r}/{m}' for r in v8roots + [v9] if os.path.isdir(f'{r}/{m}')]
        if not dirs: continue
        res = cells_for(dirs, UNT + COMP)
        if res: E[m], A[m], INFO[m] = res
    ta = lambda m, k: (A.get(m, {}).get(k) or {}).get('a')
    out = {'info': INFO, 'task_effects': {m: {k: (v['a'] if v else None) for k, v in A[m].items()} for m in A},
           'format_effects': {m: {k: (v['b'] if v else None) for k, v in A[m].items()} for m in A},
           'exponents': {m: {k: {x: round(y, 4) for x, y in v.items()} for k, v in E[m].items()} for m in E}}
    REF = {m: reference(A, m) for m in A}

    # T8 within model: compression task effects vs the untargeted reference
    t8 = {f'{m}/{c}': Z.corr_d(ta(m, c), REF[m], Z.MIN_COMMON_T) for m in MODELS7 if m in A for c in COMP}
    m8 = Z.med(t8.values())
    out['T8'] = {'by_cell': t8, 'median': m8, 'n_cells': sum(v is not None for v in t8.values()),
                 'reading': None if m8 is None else ('follows' if m8 >= 0.7 else 'does not follow' if m8 <= 0.4 else 'partly follows')}
    # T9 across models, per method
    t9 = {}
    for c in COMP:
        prs = {f'{a}|{b}': Z.corr_d(ta(a, c), ta(b, c), Z.MIN_COMMON_T) for a, b in itertools.combinations([m for m in MODELS7 if m in A], 2)}
        v = [x for x in prs.values() if x is not None]; t9[c] = {'median': Z.med(v), 'n_pairs': len(v), 'pairs': prs}
    out['T9'] = t9; out['T9_pass'] = all((t9[c]['median'] or 0) >= 0.6 for c in COMP)
    # T10 transfer of the two-factor fit from the untargeted damages
    t10 = {}
    for m in MODELS7:
        if m not in A: continue
        src = [A[m][k] for k in UNT if A[m].get(k)]
        for c in COMP:
            if not A[m].get(c) or len(src) < 2: continue
            y = A[m][c]['y']; P = {'additive': {}, 'task_only': {}, 'format_only': {}}
            for key in y:
                t, f = key.rsplit('_', 1); s = [x for x in src if t in x['a']]
                if len(s) < 2: continue
                P['additive'][key] = np.mean([x['mu'] + x['a'][t] + x['b'].get(f, 0) for x in s])
                P['task_only'][key] = np.mean([x['mu'] + x['a'][t] for x in s]); P['format_only'][key] = np.mean([x['mu'] + x['b'].get(f, 0) for x in src])
            t10[f'{m}/{c}'] = {k: Z.corr_d(P[k], y, Z.MIN_COMMON_M) for k in P}
    add = [v['additive'] for v in t10.values() if v['additive'] is not None]
    beats = [v['additive'] > max(v['task_only'] if v['task_only'] is not None else -9, v['format_only'] if v['format_only'] is not None else -9) for v in t10.values() if v['additive'] is not None]
    out['T10'] = {'by_cell': t10, 'median_additive': Z.med(add), 'frac_beats': round(float(np.mean(beats)), 3) if beats else None, 'n_cells': len(add)}
    out['T10']['pass'] = bool((out['T10']['median_additive'] or 0) >= 0.7 and (out['T10']['frac_beats'] or 0) >= 0.75)
    # T11 scale: Qwen2.5-32B
    if 'q32' in A:
        w = [Z.corr_d(ta('q32', a), ta('q32', b), Z.MIN_COMMON_T) for a, b in itertools.combinations(UNT, 2)]
        x = {f'{o}/{k}': Z.corr_d(ta('q32', k), ta(o, k), Z.MIN_COMMON_T) for o in ('q15', 'q7', 'q14') for k in UNT}
        out['T11'] = {'within_pairs': w, 'within_median': Z.med(w), 'vs_qwen': x, 'vs_qwen_median': Z.med(x.values())}
        out['T11']['pass'] = bool((out['T11']['within_median'] or 0) >= 0.6 and (out['T11']['vs_qwen_median'] or 0) >= 0.6)
    # T12 middle tasks: T8 without the extreme tasks, against a shuffled-label null of the median
    rng = np.random.default_rng(0); obs, pairs = {}, []
    for m in MODELS7:
        if m not in A: continue
        for c in COMP:
            a = ta(m, c)
            if a is None: continue
            ks = [t for t in a if t in REF[m] and t not in EXTREME]
            if len(ks) < 6: continue
            x = np.array([a[t] for t in ks]); y = np.array([REF[m][t] for t in ks]); obs[f'{m}/{c}'] = Z.pear(x, y); pairs.append((x, y))
    null = [float(np.median([Z.pear(rng.permutation(x), y) for x, y in pairs])) for _ in range(NPERM)] if pairs else []
    mo = Z.med(obs.values())
    out['T12'] = {'by_cell': {k: round(v, 3) for k, v in obs.items()}, 'median': mo, 'null_p95': round(float(np.percentile(null, 95)), 3) if null else None,
                  'n_cells': len(obs), 'min_common_tasks': 6}
    out['T12']['pass'] = bool(null and mo is not None and mo > np.percentile(null, 95))

    # ---------------- exploratory (no verdicts)
    ex = {}
    ex['between_methods'] = {m: {f'{a}|{b}': Z.corr_d(ta(m, a), ta(m, b), Z.MIN_COMMON_T) for a, b in itertools.combinations(COMP, 2)} for m in A}
    ex['format_effects_compression'] = {m: {c: (A[m][c]['b'] if A[m].get(c) else None) for c in COMP} for m in A}
    ex['per_method_vs_each_untargeted'] = {m: {c: {k: Z.corr_d(ta(m, c), ta(m, k), Z.MIN_COMMON_T) for k in UNT} for c in COMP} for m in A}
    sens = {}
    for lo, hi, mp in ((0.05, 0.95, 2), (0.02, 0.98, 2), (0.03, 0.97, 3)):   # fit sensitivity: window and minimum points (T1 and T8 medians)
        Z.LO, Z.HI = lo, hi; orig_fit = Z.fit_cell
        def fit_mp(R, base, ms, k, idx=None, _mp=mp):
            r = orig_fit(R, base, ms, k, idx)
            if r is None or _mp == 2: return r
            ex_, h, inf = r; C = inf['C']; keep = {q: v for q, v in ex_.items() if sum(Z.LO < c < Z.HI for c in C[q]) >= _mp}
            return (keep, h, inf) if len(keep) >= Z.MIN_MEASURES else None
        Z.fit_cell = fit_mp; AA = {}
        for m in MODELS7 + ['q32']:
            dirs = [f'{r}/{m}' for r in v8roots + [v9] if os.path.isdir(f'{r}/{m}')]
            res = cells_for(dirs, UNT + COMP) if dirs else None
            if res: AA[m] = res[1]
        Z.fit_cell = orig_fit; Z.LO, Z.HI = 0.03, 0.97
        tb = lambda m, k: (AA.get(m, {}).get(k) or {}).get('a')
        t1v = [Z.corr_d(tb(m, a), tb(m, b), Z.MIN_COMMON_T) for m in MODELS7 if m in AA for a, b in itertools.combinations(UNT, 2)]
        t8v = [Z.corr_d(tb(m, c), reference(AA, m), Z.MIN_COMMON_T) for m in MODELS7 if m in AA for c in COMP]
        sens[f'window {lo}-{hi}, min points {mp}'] = {'T1_median': Z.med(t1v), 'T8_median': Z.med(t8v)}
    ex['fit_sensitivity'] = sens
    out['exploratory'] = ex
    json.dump(out, open('analysis10.json', 'w'), indent=1, ensure_ascii=False)
    print('T8', out['T8']['median'], out['T8']['reading'], 'cells', out['T8']['n_cells']); print('  ', t8)
    print('T9 pass', out['T9_pass'], {c: t9[c]['median'] for c in COMP})
    print('T10 pass', out['T10']['pass'], out['T10']['median_additive'], out['T10']['frac_beats'], out['T10']['n_cells'])
    if 'T11' in out: print('T11 pass', out['T11']['pass'], out['T11']['within_median'], out['T11']['vs_qwen_median'])
    print('T12 pass', out['T12']['pass'], out['T12']['median'], 'null p95', out['T12']['null_p95'], 'cells', out['T12']['n_cells'])
    print('sensitivity', sens)

if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
