"""Addendum 18: predict the accuracy left in each measure under compression from exponents measured under random damage.
usage: python3 analyze19.py   # locked analysis -> analysis19.json
Exponents: rank-one fit (analyze9.rank1, same 0.03-0.97 window) in each random-damage cell, normalized to geometric mean 1,
averaged in log over the random damages in which the measure was fitted (at least two). Compression data are never used for them.
In each compression condition, one strength h is estimated from the other measures (leave one measure out) and the held-out
measure's accuracy is predicted as chance + (intact - chance) * exp(-n h). Baseline: the same with every exponent set to 1.
"""
import json, pickle
import numpy as np
from analyze9 import rank1, LO, HI, MIN_MEASURES
from analyze15 import MODELS, RANDOM, COMPRESS, CACHE

def cval(base, e, m):
    a0, k = base['res'][m]['acc'], base['res'][m]['chance']
    return (e['res'][m]['acc'] - k) / (a0 - k)

def random_exponents(C, m):
    cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]; ms = [x for x in sorted(C[m]['usable']) if base['res'][x]['acc'] - base['res'][x]['chance'] > 0]
    logs = {x: [] for x in ms}
    for k in RANDOM:
        doses = sorted({d for (kk, d, s) in cond if kk == k})
        if len(doses) < 3: continue
        Cm = np.array([[np.mean([cval(base, cond[(k, d, s)], x) for (kk, dd, s) in cond if kk == k and dd == d]) for d in doses] for x in ms])
        M = ((Cm > LO) & (Cm < HI)).astype(float); keep = M.sum(1) >= 2
        if keep.sum() < MIN_MEASURES: continue
        n, _ = rank1(-np.log(np.clip(Cm[keep], 1e-6, 1)), M[keep])
        for x, v in zip([x for x, q in zip(ms, keep) if q], n): logs[x].append(np.log(v))
    return {x: float(np.exp(np.mean(v))) for x, v in logs.items() if len(v) >= 2}

def predict_condition(base, e, ex):
    """leave-one-measure-out predictions of accuracy for every measure with an exponent; returns rows (measure, obs, pred_profile, pred_uniform)"""
    ms = sorted(ex); c = {x: cval(base, e, x) for x in ms}; rows = []
    for t in ms:
        use = [x for x in ms if x != t and LO < c[x] < HI]
        if len(use) < 5: continue
        L = np.array([-np.log(c[x]) for x in use]); n = np.array([ex[x] for x in use])
        h_p = float((n * L).sum() / (n * n).sum()); h_u = float(L.mean())
        a0, k = base['res'][t]['acc'], base['res'][t]['chance']
        rows.append((t, float(e['res'][t]['acc']), k + (a0 - k) * np.exp(-ex[t] * h_p), k + (a0 - k) * np.exp(-h_u)))
    return rows

def main():
    C = pickle.load(open(CACHE, 'rb')); R = {'cells': {}, 'H1': {}, 'H2': {}}
    for m in MODELS:
        ex = random_exponents(C, m); cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]
        for k in COMPRESS:
            rows, conds = [], 0
            for (kk, d, s) in sorted(cond):
                if kk != k: continue
                cm = np.median([cval(base, cond[(kk, d, s)], x) for x in ex])
                if not (0.10 <= cm <= 0.90): continue                   # informative doses only
                conds += 1; rows += predict_condition(base, cond[(kk, d, s)], ex)
            if conds == 0 or len(rows) < 20: R['cells'][f'{m}|{k}'] = None; continue
            o, p, u = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
            R['cells'][f'{m}|{k}'] = {'n_conditions': conds, 'n_points': len(rows), 'n_measures': len(ex),
                                      'mae_profile': float(np.abs(p - o).mean()), 'mae_uniform': float(np.abs(u - o).mean()),
                                      'r2_profile': float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()),
                                      'r2_uniform': float(1 - ((u - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()),
                                      'points': [[r[0], round(r[1], 4), round(float(r[2]), 4), round(float(r[3]), 4)] for r in rows]}
    for k in COMPRESS:
        cs = [c for key, c in R['cells'].items() if key.endswith('|' + k) and c]
        red = [1 - c['mae_profile'] / c['mae_uniform'] for c in cs]; better = [c['mae_profile'] < c['mae_uniform'] for c in cs]
        h1 = {'n_cells': len(cs), 'median_reduction': float(np.median(red)) if cs else None, 'frac_better': float(np.mean(better)) if cs else None,
              'median_mae_profile': float(np.median([c['mae_profile'] for c in cs])) if cs else None,
              'median_mae_uniform': float(np.median([c['mae_uniform'] for c in cs])) if cs else None,
              'median_r2_profile': float(np.median([c['r2_profile'] for c in cs])) if cs else None}
        h1['pass'] = bool(cs and len(cs) >= 4 and h1['median_reduction'] >= 0.25 and h1['frac_better'] >= 0.75); R['H1'][k] = h1
        R['H2'][k] = {'median_mae_profile': h1['median_mae_profile'], 'pass': bool(cs and len(cs) >= 4 and h1['median_mae_profile'] <= 0.10)}
    n1 = sum(R['H1'][k]['pass'] for k in COMPRESS); n2 = sum(R['H2'][k]['pass'] for k in COMPRESS)
    R['H1_verdict'] = 'COMMIT' if n1 == 3 else ('PARTIAL' if n1 == 2 else 'STOP'); R['H2_verdict'] = 'COMMIT' if n2 == 3 else ('PARTIAL' if n2 == 2 else 'STOP')
    json.dump(R, open('analysis19.json', 'w'), indent=1)
    print('H1', R['H1_verdict'], f'{n1}/3 | H2', R['H2_verdict'], f'{n2}/3')
    for k in COMPRESS:
        h = R['H1'][k]
        print(f"  {k:10s} cells {h['n_cells']}  MAE profile {h['median_mae_profile']:.3f} vs uniform {h['median_mae_uniform']:.3f}  "
              f"reduction {h['median_reduction']:.2f}  better {h['frac_better']:.2f}  R2 {h['median_r2_profile']:.2f}")

if __name__ == '__main__': main()
