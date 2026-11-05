"""Addendum 17: does item survival under random damage predict which items compression removes (knowledge level)?
usage: python3 analyze18.py   # locked analysis -> analysis18.json
"""
import json, pickle
import numpy as np
from scipy.stats import spearmanr
from analyze15 import MODELS, RANDOM, COMPRESS, CACHE, fact_countries
from analyze17 import surv

def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z held constant (Pearson on ranks, residualised on rank of z)."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    ex, ey = res(rx), res(ry)
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else None

def units(C, m, task, F15, F16, cs, countries):
    """per unit (country for FACT, item otherwise): log frequency, random-damage survival, survival under each compression method"""
    meas = f'{task}_BARE'; ok0 = C[m]['cond'][('none', 0.0, 0)]['res'][meas]['correct']
    sR = surv(C, m, meas, RANDOM); sC = {d: surv(C, m, meas, [d]) for d in COMPRESS}
    if sR is None: return None
    if task == 'FACT':
        keep = [c for c in countries if ok0[cs == c].all()]
        agg = lambda s: np.array([s[cs == c].mean() for c in keep]) if s is not None else None
        lf = np.array([np.log10(F15[c]['cooc'] + 1) for c in keep])
    else:
        ev = json.load(open(f'items_v8/{task}.json'))['eval']; idx = np.where(ok0 == 1)[0]
        agg = lambda s: s[idx] if s is not None else None
        lf = np.array([np.log10(F16[f"{task}|{ev[i]['ans']}"]['count'] + 1) for i in idx])
    return lf, agg(sR), {d: agg(s) for d, s in sC.items()}

def cell(lf, sR, sC):
    if sC is None or len(sR) < 30 or np.ptp(sR) == 0 or np.ptp(sC) == 0: return None
    return {'n': int(len(sR)), 'rho_random': float(spearmanr(sR, sC)[0]), 'rho_freq': float(spearmanr(lf, sC)[0]),
            'partial_random_given_freq': partial_spearman(sR, sC, lf)}

def summ(cells, key):
    v = [c[key] for c in cells.values() if c is not None and c[key] is not None]
    return {'n_cells': len(v), 'n_missing': len(cells) - len(v), 'median': float(np.median(v)) if v else None,
            'frac_positive': float(np.mean([x > 0 for x in v])) if v else None}

def main():
    C = pickle.load(open(CACHE, 'rb')); F15 = json.load(open('freq15.json'))['counts']; F16 = json.load(open('freq16.json'))['counts']
    cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); R = {'H1': {}, 'H2': {}, 'H3': {}, 'cells': {}}
    for task in ('FACT', 'SCIQ', 'RETR'):
        for m in MODELS:
            u = units(C, m, task, F15, F16, cs, countries)
            for d in COMPRESS: R['cells'][f'{m}|{task}|{d}'] = cell(u[0], u[1], u[2][d]) if u else None
    for d in COMPRESS:
        k = {key: c for key, c in R['cells'].items() if key.endswith('|' + d) and '|RETR|' not in key}
        h = summ(k, 'rho_random'); h['pass'] = bool(h['median'] is not None and h['median'] >= 0.30 and h['frac_positive'] >= 0.75)
        h['freq_only'] = summ(k, 'rho_freq'); R['H1'][d] = h
        p = summ(k, 'partial_random_given_freq'); p['pass'] = bool(p['median'] is not None and p['median'] >= 0.15 and p['frac_positive'] >= 0.75); R['H2'][d] = p
        R['H3'][d] = summ({key: c for key, c in R['cells'].items() if key.endswith('|' + d) and '|RETR|' in key}, 'rho_random')
    n1 = sum(R['H1'][d]['pass'] for d in COMPRESS); n2 = sum(R['H2'][d]['pass'] for d in COMPRESS)
    R['H1_verdict'] = 'COMMIT' if n1 == 3 else ('PARTIAL' if n1 == 2 else 'STOP')
    R['H2_verdict'] = 'COMMIT' if n2 == 3 else ('PARTIAL' if n2 == 2 else 'STOP')
    json.dump(R, open('analysis18.json', 'w'), indent=1)
    print('H1', R['H1_verdict'], f'{n1}/3', '| H2', R['H2_verdict'], f'{n2}/3')
    for d in COMPRESS:
        h, p, f, r = R['H1'][d], R['H2'][d], R['H1'][d]['freq_only'], R['H3'][d]
        print(f"  {d:10s} random->compression median {h['median']:.3f} pos {h['frac_positive']:.2f} cells {h['n_cells']} (missing {h['n_missing']}) | "
              f"freq only {f['median']:.3f} | partial given freq {p['median']:.3f} pos {p['frac_positive']:.2f} | RETR {r['median']}")

if __name__ == '__main__': main()
