"""Locked analysis for PROTOCOL_ADDENDUM_8. Written with the protocol, before any result of this batch existed.
  python3 analyze9.py <run_root> [<run_root2> ...]      -> analysis9.json + printed verdicts
Each run root holds one directory per model key (q15, q7, ...), as written by atlas_full.py.
"""
import sys, os, json, gzip, glob, itertools
import numpy as np

TASKS = ['ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'COPY', 'SCIQ', 'BOOLQ', 'ARC', 'CSQA', 'MMLU_CALC', 'MMLU_MEM', 'HELLASWAG', 'PIQA', 'WINOGRANDE', 'OBQA']
FORMATS = ['CHOICE', 'BARE', 'JSON', 'TWICE']
UNT = ['noise', 'zero', 'neuron', 'ld_rand']; ALL = UNT + ['ld_late', 'qmix']
MODELS = ['q15', 'q7', 'q14', 'llama', 'mis', 'gemma', 'phi']
EXT_NAME = {'q15': 'Qwen-1.5B', 'q7': 'Qwen-7B', 'q14': 'Qwen-14B', 'llama': 'Llama-3B', 'mis': 'Mistral-7B', 'gemma': 'Gemma-9B', 'phi': 'Phi-3.5'}
# addendum-7 measure -> this batch (T6). BARE takes the addendum-7 GEN measure where one exists, F0 only for COPY.
EXT_MAP = {f'{t}_GEN': f'{t}_BARE' for t in ('ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR')}
EXT_MAP.update({f'{t}_CHOICE': f'{t}_CHOICE' for t in ('ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR')})
EXT_MAP.update({'COPY_F0': 'COPY_BARE', **{f'{t}_F2': f'{t}_JSON' for t in ('FACT', 'ARITH1', 'COPY')}})   # F5 dropped: TWICE wording changed after smoke
LO, HI, MIN_MEASURES, MIN_TASKS, MIN_COMMON_M, MIN_COMMON_T = 0.03, 0.97, 12, 8, 10, 8

def pear(a, b): return float(np.corrcoef(a, b)[0, 1])
def med(v): v = [x for x in v if x is not None]; return round(float(np.median(v)), 3) if v else None
def rank1(L, M, it=500):   # identical to analyze8.rank1
    n = np.ones(L.shape[0]); h = np.ones(L.shape[1])
    for _ in range(it):
        h = np.array([(M[:, j] * n * L[:, j]).sum() / max((M[:, j] * n * n).sum(), 1e-12) for j in range(L.shape[1])])
        n = np.array([(M[i] * h * L[i]).sum() / max((M[i] * h * h).sum(), 1e-12) for i in range(L.shape[0])])
    s = np.exp(np.mean(np.log(n[n > 0]))); return n / s, h * s

# ---------------------------------------------------------------- loading
def load_model(dirs):
    R = {}
    for d in dirs:
        for f in glob.glob(f'{d}/*.json.gz'):
            with gzip.open(f, 'rt') as fh: x = json.load(fh)
            c = x['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = {'res': {m: np.array(v['correct']) for m, v in x['res'].items()}, 'chance': {m: v['chance'] for m, v in x['res'].items()}, 'probe': x['probe']}
    use = None
    for d in dirs:
        if os.path.exists(f'{d}/usable.json'): use = json.load(open(f'{d}/usable.json'))
    return R, use

def fit_cell(R, base, ms, k, idx=None):
    """exponents for one model x damage; idx = item subset (for split halves). Returns ({measure: n}, {dose: h}, info) or None."""
    doses = sorted({x[1] for x in R if x[0] == k})
    if len(doses) < 3: return None
    seeds = sorted({x[2] for x in R if x[0] == k})
    def acc(e, m): v = e['res'][m]; return float(v.mean() if idx is None else v[idx].mean())
    C = np.array([[np.mean([(acc(R[(k, x, s)], m) - base['chance'][m]) / (acc(base, m) - base['chance'][m]) for s in seeds if (k, x, s) in R]) for x in doses] for m in ms])
    M = ((C > LO) & (C < HI)).astype(float); keep = M.sum(1) >= 2
    if keep.sum() < MIN_MEASURES: return None
    n, h = rank1(-np.log(np.clip(C[keep], 1e-6, 1)), M[keep]); P = np.exp(-np.outer(n, h)); mm = M[keep] > 0
    r2 = float(1 - ((C[keep][mm] - P[mm]) ** 2).sum() / ((C[keep][mm] - C[keep][mm].mean()) ** 2).sum())
    kl = {x: float(np.mean([R[(k, x, s)]['probe']['kl'] for s in seeds if (k, x, s) in R])) for x in doses}
    return {m: float(v) for m, v in zip([m for m, q in zip(ms, keep) if q], n)}, dict(zip(doses, h.tolist())), {'n_fitted': int(keep.sum()), 'R2': round(r2, 3), 'doses': doses, 'kl': kl, 'C': {m: c.tolist() for m, c in zip(ms, C)}}

def additive(ex):
    """least-squares log n = mu + a_task + b_format over tasks with >=3 formats; missing cells allowed."""
    if ex is None: return None
    cells = {(t, f): np.log(ex[f'{t}_{f}']) for t in TASKS for f in FORMATS if f'{t}_{f}' in ex}
    tasks = [t for t in TASKS if sum((t, f) in cells for f in FORMATS) >= 3]
    if len(tasks) < MIN_TASKS: return None
    cells = {k: v for k, v in cells.items() if k[0] in tasks}; fm = [f for f in FORMATS if any(k[1] == f for k in cells)]
    mu = np.mean(list(cells.values())); a = {t: 0.0 for t in tasks}; b = {f: 0.0 for f in fm}
    for _ in range(200):
        a = {t: np.mean([v - mu - b[f] for (tt, f), v in cells.items() if tt == t]) for t in tasks}; ma = np.mean(list(a.values())); a = {t: x - ma for t, x in a.items()}
        b = {f: np.mean([v - mu - a[t] for (t, ff), v in cells.items() if ff == f]) for f in fm}; mb = np.mean(list(b.values())); b = {f: x - mb for f, x in b.items()}
        mu = np.mean([v - a[t] - b[f] for (t, f), v in cells.items()])
    fit = {k: mu + a[k[0]] + b[k[1]] for k in cells}
    return {'mu': float(mu), 'a': {t: float(x) for t, x in a.items()}, 'b': {f: float(x) for f, x in b.items()}, 'y': {f'{t}_{f}': float(v) for (t, f), v in cells.items()},
            'resid': {f'{t}_{f}': float(v - fit[(t, f)]) for (t, f), v in cells.items()}}

def corr_d(x, y, mn):
    if x is None or y is None: return None
    ks = [k for k in x if k in y]
    return round(pear([x[k] for k in ks], [y[k] for k in ks]), 3) if len(ks) >= mn else None

# ---------------------------------------------------------------- main
def main(roots):
    E, A, H, INFO, AH, USE = {}, {}, {}, {}, {}, {}
    for m in MODELS:
        dirs = [f'{r}/{m}' for r in roots if os.path.isdir(f'{r}/{m}')]
        if not dirs: continue
        R, use = load_model(dirs)
        if ('none', 0, 0) not in R or use is None: continue
        base = R[('none', 0, 0)]; ms = sorted(use['usable']); USE[m] = use
        nitem = len(next(iter(base['res'].values()))); halves = [np.arange(0, nitem, 2), np.arange(1, nitem, 2)]
        E[m], A[m], H[m], INFO[m], AH[m] = {}, {}, {}, {}, {}
        for k in ALL:
            r = fit_cell(R, base, ms, k)
            if r is None: INFO[m][k] = 'not measurable'; continue
            E[m][k], H[m][k], INFO[m][k] = r; A[m][k] = additive(E[m][k])
            AH[m][k] = []
            for idx in halves:
                rh = fit_cell(R, base, ms, k, idx); AH[m][k].append(additive(rh[0]) if rh else None)
    out = {'info': {m: {k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items() if kk != 'C'}) for k, v in INFO[m].items()} for m in INFO},
           'usable_counts': {m: {'measures': len(u['usable']), 'tabled_tasks_intact': len(u['tabled_tasks'])} for m, u in USE.items()},
           'task_effects': {m: {k: (v['a'] if v else None) for k, v in A[m].items()} for m in A}, 'format_effects': {m: {k: (v['b'] if v else None) for k, v in A[m].items()} for m in A},
           'exponents': {m: {k: {x: round(y, 4) for x, y in v.items()} for k, v in E[m].items()} for m in E}}
    ta = lambda m, k: (A[m].get(k) or {}).get('a') if m in A else None

    # T7 reliability of task effects (item split halves)
    rel_t = {f'{m}/{k}': corr_d((AH[m][k][0] or {}).get('a'), (AH[m][k][1] or {}).get('a'), MIN_COMMON_T) for m in AH for k in UNT if k in AH[m]}
    out['T7'] = {'by_cell': rel_t, 'median': med(rel_t.values())}; out['T7']['pass'] = bool((out['T7']['median'] or 0) >= 0.8)
    # T1 within model, task level
    t1 = {}; allv = []
    for m in A:
        prs = {f'{a}-{b}': corr_d(ta(m, a), ta(m, b), MIN_COMMON_T) for a, b in itertools.combinations(UNT, 2)}
        v = [x for x in prs.values() if x is not None]; t1[m] = {'pairs': prs, 'median': med(v)}; allv += v
    n6 = sum((x['median'] or 0) >= 0.6 for x in t1.values())
    out['T1'] = {'by_model': t1, 'overall_median': med(allv), 'n_pairs': len(allv), 'models_median_ge_0.6': n6, 'n_models': len(t1)}
    out['T1']['pass'] = bool((out['T1']['overall_median'] or 0) >= 0.7 and n6 >= 6)
    # T2 across models, task level
    t2 = {}
    for k in UNT:
        prs = {f'{a}|{b}': corr_d(ta(a, k), ta(b, k), MIN_COMMON_T) for a, b in itertools.combinations(list(A), 2)}
        v = [x for x in prs.values() if x is not None]; t2[k] = {'median': med(v), 'min': min(v) if v else None, 'n_pairs': len(v), 'pairs': prs}
    out['T2'] = t2; out['T2_pass'] = all((t2[k]['median'] or 0) >= 0.6 for k in UNT)
    # T3 additive prediction of a held-out damage; T4 measure-level prediction and residual stability
    t3 = {}
    for m in A:
        for d in UNT:
            if not A[m].get(d): continue
            others = [o for o in UNT if o != d and A[m].get(o)]
            if len(others) < 3: continue
            y = A[m][d]['y']; P = {'additive': {}, 'task_only': {}, 'format_only': {}, 'measure': {}}
            for key in y:
                t, f = key.rsplit('_', 1)
                pa = [A[m][o]['mu'] + A[m][o]['a'][t] + A[m][o]['b'].get(f, 0) for o in others if t in A[m][o]['a']]
                if len(pa) < 2: continue
                P['additive'][key] = np.mean(pa); P['task_only'][key] = np.mean([A[m][o]['mu'] + A[m][o]['a'][t] for o in others if t in A[m][o]['a']])
                P['format_only'][key] = np.mean([A[m][o]['mu'] + A[m][o]['b'].get(f, 0) for o in others])
                pm = [np.log(E[m][o][key]) for o in others if key in E[m][o]]
                if pm: P['measure'][key] = np.mean(pm)
            t3[f'{m}/{d}'] = {k: corr_d(P[k], y, MIN_COMMON_M) for k in P}
    add = [v['additive'] for v in t3.values() if v['additive'] is not None]
    beats = [v['additive'] > max(v['task_only'] if v['task_only'] is not None else -9, v['format_only'] if v['format_only'] is not None else -9) for v in t3.values() if v['additive'] is not None]
    out['T3'] = {'by_case': t3, 'median_additive': med(add), 'frac_additive_beats_single_factor': round(float(np.mean(beats)), 3) if beats else None, 'n_cases': len(add)}
    out['T3']['pass'] = bool((out['T3']['median_additive'] or 0) >= 0.7 and (out['T3']['frac_additive_beats_single_factor'] or 0) >= 0.75)
    rel_r = {f'{m}/{k}': corr_d((AH[m][k][0] or {}).get('resid'), (AH[m][k][1] or {}).get('resid'), MIN_COMMON_M) for m in AH for k in UNT if k in AH[m]}
    dis = []; raw = []
    for m in A:
        for a, b in itertools.combinations(UNT, 2):
            if not (A[m].get(a) and A[m].get(b)): continue
            r = corr_d(A[m][a]['resid'], A[m][b]['resid'], MIN_COMMON_M); ra, rb = rel_r.get(f'{m}/{a}'), rel_r.get(f'{m}/{b}')
            if r is None: continue
            raw.append(r)
            if ra is not None and rb is not None and ra > 0.2 and rb > 0.2: dis.append(r / np.sqrt(ra * rb))
    dm = med(dis)
    out['T4'] = {'resid_reliability': rel_r, 'resid_reliability_median': med(rel_r.values()), 'cross_damage_resid_median': med(raw), 'disattenuated_median': dm, 'n_disattenuated': len(dis),
                 'measure_vs_additive_prediction': {'median_measure': med([v['measure'] for v in t3.values()]), 'median_additive': med(add)},
                 'reading': None if dm is None else ('measure-specific part stable' if dm >= 0.6 else 'measure-specific part not stable' if dm <= 0.3 else 'undecided')}
    # T5 position control
    t5 = {}
    for m in A:
        c = corr_d(ta(m, 'noise'), ta(m, 'ld_late'), MIN_COMMON_T)
        if c is not None and t1[m]['median'] is not None: t5[m] = {'noise_vs_ld_late': c, 'untargeted_median': t1[m]['median'], 'holds': bool(c < t1[m]['median'])}
    out['T5'] = t5; out['T5_pass'] = sum(v['holds'] for v in t5.values()) >= 5
    # T6 reproduction of addendum 7
    ext = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext_analysis8.json')))['n']; t6 = {}
    for m in E:
        for k in UNT:
            old = ext.get(EXT_NAME[m], {}).get(k); new = E[m].get(k)
            if not old or not new: continue
            pairs = [(np.log(old[o]), np.log(new[n])) for o, n in EXT_MAP.items() if o in old and n in new]
            t6[f'{m}/{k}'] = round(pear(*zip(*pairs)), 3) if len(pairs) >= 8 else None
    out['T6'] = {'by_cell': t6, 'median': med(t6.values())}; out['T6']['pass'] = bool((out['T6']['median'] or 0) >= 0.8)

    # ---------------- exploratory (no verdicts)
    ex = {}
    ex['qmix_vs_untargeted_task_level'] = {m: {k: corr_d(ta(m, 'qmix'), ta(m, k), MIN_COMMON_T) for k in UNT} for m in A}
    ex['h_vs_kl_spearman'] = {}
    for m in H:
        for k in H[m]:
            ds = INFO[m][k]['doses']; h = [H[m][k][x] for x in ds]; kl = [INFO[m][k]['kl'][x] for x in ds]
            ex['h_vs_kl_spearman'][f'{m}/{k}'] = round(pear(np.argsort(np.argsort(h)), np.argsort(np.argsort(kl))), 3) if len(ds) >= 3 else None
    d50 = {}
    for m in INFO:   # half-dose on the probe-KL axis: KL at which the seed-averaged corrected score first crosses 0.5 (log-linear interpolation)
        d50[m] = {}
        for k, inf in INFO[m].items():
            if isinstance(inf, str): continue
            kls = np.array([0.0] + [inf['kl'][x] for x in inf['doses']]); d50[m][k] = {}
            for meas, c in inf['C'].items():
                y = np.minimum.accumulate(np.array([1.0] + c))
                if y[-1] > 0.5: continue
                j = int(np.argmax(y <= 0.5)); x0, x1, y0, y1 = kls[j - 1], kls[j], y[j - 1], y[j]
                d50[m][k][meas] = float(x0 + (x1 - x0) * (y0 - 0.5) / max(y0 - y1, 1e-9))
    ex['exponent_vs_negD50_within_cell'] = {f'{m}/{k}': corr_d({q: np.log(v) for q, v in E[m][k].items()}, {q: -np.log(v) for q, v in d50[m].get(k, {}).items() if v > 0}, MIN_COMMON_M) for m in E for k in E[m]}
    ex['T1_with_D50_measure_level'] = {m: med([corr_d({q: np.log(v) for q, v in d50[m].get(a, {}).items() if v > 0}, {q: np.log(v) for q, v in d50[m].get(b, {}).items() if v > 0}, MIN_COMMON_M) for a, b in itertools.combinations(UNT, 2)]) for m in d50}
    ex['T1_with_exponent_measure_level'] = {m: med([corr_d({q: np.log(v) for q, v in E[m].get(a, {}).items()}, {q: np.log(v) for q, v in E[m].get(b, {}).items()}, MIN_COMMON_M) for a, b in itertools.combinations(UNT, 2)]) for m in E}
    out['exploratory'] = ex
    json.dump(out, open('analysis9.json', 'w'), indent=1, ensure_ascii=False)
    print('INFO'); [print('  ', m, {k: (v if isinstance(v, str) else (v['n_fitted'], v['R2'], len(v['doses']))) for k, v in INFO[m].items()}) for m in INFO]
    print('T7 reliability median', out['T7']['median'], 'pass', out['T7']['pass'])
    print('T1 pass', out['T1']['pass'], 'overall', out['T1']['overall_median'], 'pairs', out['T1']['n_pairs'], 'models>=0.6', n6, '/', len(t1)); [print('  ', m, v['median'], v['pairs']) for m, v in t1.items()]
    print('T2 pass', out['T2_pass']); [print('  ', k, v['median'], 'min', v['min'], 'n', v['n_pairs']) for k, v in t2.items()]
    print('T3 pass', out['T3']['pass'], 'median additive', out['T3']['median_additive'], 'beats single factor', out['T3']['frac_additive_beats_single_factor'], 'n', out['T3']['n_cases'])
    print('T4', {k: v for k, v in out['T4'].items() if k != 'resid_reliability'})
    print('T5 pass', out['T5_pass'], t5); print('T6 pass', out['T6']['pass'], out['T6']['median'], t6)

if __name__ == '__main__': main(sys.argv[1:])
