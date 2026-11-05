"""Exploratory checks requested in review (not preregistered), existing data only -> analysis27_explore.json
(1) Empirical transfer baseline for accuracy prediction (no rank-one model): for a target compression condition, the mean
    relative score of the four calibration tasks is computed; in every other model (target family excluded), each measure's
    relative score is interpolated, over that model's random-damage conditions, at the same calibration score; the mean over
    models is the prediction. Compared with the own profile (P), the transferred profile (B2) and uniform decline (U) under FIX.
(2) Window sensitivity of the shared order: within-model task-effect correlations across random damages, and compression
    against the random-damage reference, with fitting windows 0.03-0.97 (paper), 0.05-0.95, 0.10-0.90, and with at least
    three points per measure.
(3) One model per family (q7, llama, mis, gemma, phi): across-model task-effect correlations under each damage.
"""
import json, pickle, itertools
import numpy as np
from analyze9 import rank1, additive, MIN_MEASURES, MIN_COMMON_T
from analyze15 import MODELS, RANDOM, COMPRESS, CACHE
from analyze19 import cval
from analyze25 import prepare, sources, predict, calibration_tasks, FAMILY, base_of, split

def seed_mean(C, m, k):
    cond = C[m]['cond']; out = {}
    for (kk, d, s) in cond:
        if kk == k: out.setdefault(d, []).append((kk, d, s))
    return out

# ---------------------------------------------------------------- (1) empirical transfer
def curves(C, o, cal):
    """for model o: list of (calibration score, {measure: relative score}) over its random-damage conditions, seed-averaged"""
    base = base_of(C, o); ms = [x for x in C[o]['usable'] if base['res'][x]['acc'] - base['res'][x]['chance'] > 0]
    calm = [x for x in ms if split(x)[0] in cal]; pts = []
    for k in RANDOM:
        for d, keys in seed_mean(C, o, k).items():
            c = {x: float(np.mean([cval(base, C[o]['cond'][key], x) for key in keys])) for x in ms}
            pts.append((float(np.mean([np.clip(c[x], 0, 1) for x in calm])), c))
    pts.sort(key=lambda p: p[0]); return pts

def empirical(C, cal):
    MG, AD, EX = prepare(C); CV = {o: curves(C, o, cal) for o in MODELS}; cells = {}
    for m in MODELS:
        others = [o for o in MODELS if o != m and FAMILY[o] != FAMILY[m]]
        S, sF, _ = sources(C, m, MG, AD, EX, [o for o in MODELS if o != m]); _, sA, _ = sources(C, m, MG, AD, EX, others)
        src = {'P': sF['P'], 'B2': sA['B2'], 'U': sF['U']}; cond, base = C[m]['cond'], base_of(C, m)
        calm = [x for x in S if split(x)[0] in cal]
        for k in COMPRESS:
            rows = []
            for key in sorted(cond):
                if key[0] != k or not (0.10 <= np.median([cval(base, cond[key], x) for x in S]) <= 0.90): continue
                e = cond[key]; rr = predict(base, e, S, src, 'FIX', cal)
                if not rr: continue
                g = float(np.mean([np.clip(cval(base, e, x), 0, 1) for x in calm]))
                for t, obs, pr in rr:
                    preds = []
                    for o in others:
                        xs = [p[0] for p in CV[o] if t in p[1]]; ys = [p[1][t] for p in CV[o] if t in p[1]]
                        if len(xs) >= 3: preds.append(float(np.interp(g, xs, ys)))
                    if not preds: continue
                    a0, kk = base['res'][t]['acc'], base['res'][t]['chance']
                    rows.append((obs, pr['P'], pr['B2'], pr['U'], kk + (a0 - kk) * float(np.mean(preds))))
            if len(rows) < 20: continue
            R = np.array(rows); cells[f'{m}|{k}'] = {s: float(np.abs(R[:, i] - R[:, 0]).mean()) for i, s in ((1, 'P'), (2, 'B2'), (3, 'U'), (4, 'E'))}
    summ = {k: {s: float(np.median([c[s] for key, c in cells.items() if key.endswith(k)])) for s in ('P', 'B2', 'U', 'E')} for k in COMPRESS}
    for k in COMPRESS:
        cs = [c for key, c in cells.items() if key.endswith(k)]
        summ[k]['E_better_than_U'] = f"{sum(c['E'] < c['U'] for c in cs)}/{len(cs)}"
        summ[k]['B2_better_than_E'] = f"{sum(c['B2'] < c['E'] for c in cs)}/{len(cs)}"
    return cells, summ

# ---------------------------------------------------------------- (2) window sensitivity, (3) one model per family
def cell_exponents(C, m, k, lo, hi, minpts):
    cond, base = C[m]['cond'], base_of(C, m); ms = sorted(C[m]['usable'])
    doses = sorted({d for (kk, d, s) in cond if kk == k})
    if len(doses) < 3: return None
    Cm = np.array([[np.mean([cval(base, cond[(k, d, s)], x) for (kk, dd, s) in cond if kk == k and dd == d]) for d in doses] for x in ms])
    M = ((Cm > lo) & (Cm < hi)).astype(float); keep = M.sum(1) >= minpts
    if keep.sum() < MIN_MEASURES: return None
    n, _ = rank1(-np.log(np.clip(Cm[keep], 1e-6, 1)), M[keep])
    return dict(zip([x for x, q in zip(ms, keep) if q], n.tolist()))

def pear(x, y):
    if not x or not y: return None
    ks = [t for t in x if t in y]
    return float(np.corrcoef([x[t] for t in ks], [y[t] for t in ks])[0, 1]) if len(ks) >= MIN_COMMON_T else None

def order_stats(C, lo, hi, minpts, models=MODELS):
    A = {m: {k: additive(cell_exponents(C, m, k, lo, hi, minpts)) for k in RANDOM + COMPRESS} for m in models}
    ta = lambda m, k: (A[m][k] or {}).get('a')
    within = [r for m in models for a, b in itertools.combinations(RANDOM, 2) if (r := pear(ta(m, a), ta(m, b))) is not None]
    comp = []
    for m in models:
        ref = {}
        for k in RANDOM:
            for t, v in (ta(m, k) or {}).items(): ref.setdefault(t, []).append(v)
        ref = {t: float(np.mean(v)) for t, v in ref.items() if len(v) >= 2}
        comp += [r for k in COMPRESS if (r := pear(ref, ta(m, k))) is not None]
    across = {k: [r for a, b in itertools.combinations(models, 2) if (r := pear(ta(a, k), ta(b, k))) is not None] for k in RANDOM + COMPRESS}
    return {'within_random_median': float(np.median(within)), 'n_within': len(within), 'compression_vs_random_median': float(np.median(comp)), 'n_comp': len(comp),
            'across_models_median': {k: (float(np.median(v)) if v else None, len(v)) for k, v in across.items()}}

def main():
    C = pickle.load(open(CACHE, 'rb')); cal, _ = calibration_tasks(C); out = {}
    cells, summ = empirical(C, cal); out['empirical_transfer'] = {'cells': cells, 'summary': summ}
    print('(1) empirical transfer, median MAE (P own, B2 transferred, U uniform, E empirical):')
    for k in COMPRESS: print('  ', k, {s: (round(v, 3) if isinstance(v, float) else v) for s, v in summ[k].items()})
    out['window'] = {}
    for lo, hi, mp in ((.03, .97, 2), (.05, .95, 2), (.10, .90, 2), (.03, .97, 3)):
        r = order_stats(C, lo, hi, mp); out['window'][f'{lo}-{hi}|min{mp}'] = r
        print(f'(2) window {lo}-{hi}, min points {mp}: within random {r["within_random_median"]:.3f} (n={r["n_within"]}), compression vs random {r["compression_vs_random_median"]:.3f} (n={r["n_comp"]})')
    fam = ['q7', 'llama', 'mis', 'gemma', 'phi']; r = order_stats(C, .03, .97, 2, fam); out['one_per_family'] = r
    allm = out['window']['0.03-0.97|min2']['across_models_median']
    print('(3) across models, all eight vs one per family:', {k: (round(allm[k][0], 3) if allm[k][0] else None, round(r['across_models_median'][k][0], 3) if r['across_models_median'][k][0] else None, r['across_models_median'][k][1]) for k in RANDOM + COMPRESS})
    json.dump(out, open('analysis27_explore.json', 'w'), indent=1)

if __name__ == '__main__': main()
