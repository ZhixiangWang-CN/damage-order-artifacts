"""Development for addendum 19 (exploratory, random damage only; compression data are not read).
Chooses how to predict accuracy from exponents by predicting each held-out random damage from the other three.
usage: python3 analyze20_dev.py   -> analysis20_dev.json
Variants: exponent source (mean of per-cell exponents | two-axis smoothed | half-way between, in log) x
          strength estimate (least squares on -log c inside the window | least squares on c over all other measures).
"""
import json, pickle
import numpy as np
from scipy.optimize import minimize_scalar
from analyze9 import rank1, additive, LO, HI, MIN_MEASURES
from analyze15 import MODELS, RANDOM, CACHE
from analyze19 import cval

def cell_exponents(C, m, k):
    cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]
    ms = [x for x in sorted(C[m]['usable']) if base['res'][x]['acc'] - base['res'][x]['chance'] > 0]
    doses = sorted({d for (kk, d, s) in cond if kk == k})
    if len(doses) < 3: return None
    Cm = np.array([[np.mean([cval(base, cond[(k, d, s)], x) for (kk, dd, s) in cond if kk == k and dd == d]) for d in doses] for x in ms])
    M = ((Cm > LO) & (Cm < HI)).astype(float); keep = M.sum(1) >= 2
    if keep.sum() < MIN_MEASURES: return None
    n, _ = rank1(-np.log(np.clip(Cm[keep], 1e-6, 1)), M[keep])
    return dict(zip([x for x, q in zip(ms, keep) if q], n))

def exponents(cells, source):
    logs = {}
    for ex in cells:
        for x, v in ex.items(): logs.setdefault(x, []).append(np.log(v))
    mean = {x: float(np.mean(v)) for x, v in logs.items() if len(v) >= 2}
    if source == 'mean' or not mean: return {x: np.exp(v) for x, v in mean.items()}
    A = additive({x: np.exp(v) for x, v in mean.items()})
    if A is None: return {x: np.exp(v) for x, v in mean.items()}
    sm = {x: A['mu'] + A['a'][x.rsplit('_', 1)[0]] + A['b'][x.rsplit('_', 1)[1]] for x in mean if x.rsplit('_', 1)[0] in A['a'] and x.rsplit('_', 1)[1] in A['b']}
    if source == 'additive': return {x: float(np.exp(sm.get(x, mean[x]))) for x in mean}
    return {x: float(np.exp(.5 * mean[x] + .5 * sm.get(x, mean[x]))) for x in mean}   # 'shrink'

def predict(base, e, ex, how):
    ms = sorted(ex); c = {x: cval(base, e, x) for x in ms}; rows = []
    for t in ms:
        others = [x for x in ms if x != t]
        inw = [x for x in others if LO < c[x] < HI]
        if len(inw) < 5: continue
        if how == 'log':
            L = np.array([-np.log(c[x]) for x in inw]); n = np.array([ex[x] for x in inw]); h = (n * L).sum() / (n * n).sum()
        else:
            cc = np.clip(np.array([c[x] for x in others]), 0, 1); n = np.array([ex[x] for x in others])
            h = minimize_scalar(lambda hh: ((np.exp(-n * hh) - cc) ** 2).sum(), bounds=(0, 50), method='bounded').x
        a0, k = base['res'][t]['acc'], base['res'][t]['chance']
        rows.append((float(e['res'][t]['acc']), k + (a0 - k) * np.exp(-ex[t] * h)))
    return rows

def main():
    C = pickle.load(open(CACHE, 'rb')); V = [(s, h) for s in ('mean', 'additive', 'shrink') for h in ('log', 'acc')]
    out = {f'{s}|{h}': {} for s, h in V}
    for m in MODELS:
        cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]
        per = {k: cell_exponents(C, m, k) for k in RANDOM}
        for k in RANDOM:
            if per[k] is None: continue
            train = [per[j] for j in RANDOM if j != k and per[j] is not None]
            for s, h in V:
                ex = exponents(train, s); rows = []
                for key in sorted(cond):
                    if key[0] != k: continue
                    if not (0.10 <= np.median([cval(base, cond[key], x) for x in ex]) <= 0.90): continue
                    rows += predict(base, cond[key], ex, h)
                if len(rows) >= 20:
                    o, p = np.array(rows).T; out[f'{s}|{h}'][f'{m}|{k}'] = float(np.abs(p - o).mean())
    json.dump(out, open('analysis20_dev.json', 'w'), indent=1)
    for v, cells in out.items():
        print(f'{v:16s} cells {len(cells)}  median MAE {np.median(list(cells.values())):.3f}')

if __name__ == '__main__': main()
