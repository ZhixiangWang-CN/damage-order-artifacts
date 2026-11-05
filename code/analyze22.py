"""Addendum 20: are the knowledge axis and the format axis separable?
usage: python3 analyze22.py   # locked analysis -> analysis22.json
Survival of a unit (country for capitals, item for SciQ) in a measure = mean correctness over all damaged conditions of a model
(all seven damages, all doses and seeds), as in addenda 14-16; units the intact model answered correctly in that measure.
H1 knowledge does not depend on the format: Spearman(log answer frequency, survival) in BARE, JSON and TWICE.
H2 format does not depend on knowledge: units correct in all three formats, split at the median log frequency within a cell;
    survival order BARE >= JSON >= TWICE inside the rare half and inside the common half.
"""
import json, pickle
import numpy as np
from scipy.stats import spearmanr
from analyze15 import MODELS, DMG, CACHE, fact_countries

FMT = ['BARE', 'JSON', 'TWICE']; TASKS = ['FACT', 'SCIQ']

def unit_table(C, m, task, F15, F16, cs, countries):
    """per format: (intact-correct mask over units, survival over units); plus log frequency over units"""
    cond = C[m]['cond']; base = cond[('none', 0.0, 0)]['res']; out = {}
    if task == 'FACT':
        units = countries; lf = np.array([np.log10(F15[c]['cooc'] + 1) for c in units])
    else:
        ev = json.load(open(f'items_v8/{task}.json'))['eval']; units = list(range(len(ev))); lf = np.array([np.log10(F16[f"{task}|{it['ans']}"]['count'] + 1) for it in ev])
    for f in FMT:
        meas = f'{task}_{f}'
        if meas not in base or meas not in C[m]['usable']: continue
        ok0 = np.array(base[meas]['correct']); s = np.mean([cond[k]['res'][meas]['correct'] for k in cond if k[0] in DMG], 0)
        if task == 'FACT':
            ok = np.array([ok0[cs == c].all() for c in units]); sv = np.array([s[cs == c].mean() for c in units])
        else:
            ok = ok0 == 1; sv = s
        out[f] = (ok, sv)
    return lf, out

def main():
    C = pickle.load(open(CACHE, 'rb')); F15 = json.load(open('freq15.json'))['counts']; F16 = json.load(open('freq16.json'))['counts']
    cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); R = {'H1': {f: {} for f in FMT}, 'H2': {}}
    for m in MODELS:
        for t in TASKS:
            lf, T = unit_table(C, m, t, F15, F16, cs, countries)
            for f in FMT:
                if f not in T: R['H1'][f][f'{m}|{t}'] = None; continue
                ok, sv = T[f]
                R['H1'][f][f'{m}|{t}'] = float(spearmanr(lf[ok], sv[ok])[0]) if ok.sum() >= 30 and np.ptp(sv[ok]) > 0 else None
            if not all(f in T for f in FMT): R['H2'][f'{m}|{t}'] = None; continue
            ok = T['BARE'][0] & T['JSON'][0] & T['TWICE'][0]
            if ok.sum() < 30: R['H2'][f'{m}|{t}'] = None; continue
            med = np.median(lf[ok]); halves = {'rare': ok & (lf < med), 'common': ok & (lf >= med)}
            S = {h: {f: float(T[f][1][mask].mean()) for f in FMT} for h, mask in halves.items()}
            R['H2'][f'{m}|{t}'] = {'n': int(ok.sum()), 'survival': S,
                                   'full_order': all(S[h]['BARE'] >= S[h]['JSON'] >= S[h]['TWICE'] for h in S),
                                   'twice_lowest': all(S[h]['TWICE'] <= min(S[h]['BARE'], S[h]['JSON']) for h in S)}
    for f in FMT:
        v = [x for x in R['H1'][f].values() if x is not None]
        R['H1'][f] = {'cells': R['H1'][f], 'n_cells': len(v), 'median': float(np.median(v)) if v else None, 'frac_positive': float(np.mean([x > 0 for x in v])) if v else None}
        R['H1'][f]['pass'] = bool(len(v) >= 8 and R['H1'][f]['median'] >= 0.20 and R['H1'][f]['frac_positive'] >= 0.75)
    n1 = sum(R['H1'][f]['pass'] for f in FMT); R['H1_verdict'] = 'COMMIT' if n1 == 3 else ('PARTIAL' if n1 == 2 and R['H1']['BARE']['pass'] else 'STOP')
    h2 = [c for c in R['H2'].values() if c]; fo = float(np.mean([c['full_order'] for c in h2])) if h2 else 0; tl = float(np.mean([c['twice_lowest'] for c in h2])) if h2 else 0
    R['H2_summary'] = {'n_cells': len(h2), 'frac_full_order': fo, 'frac_twice_lowest': tl}
    R['H2_verdict'] = 'COMMIT' if len(h2) >= 8 and fo >= 0.75 else ('PARTIAL' if len(h2) >= 8 and tl >= 0.75 else 'STOP')
    json.dump(R, open('analysis22.json', 'w'), indent=1)
    print('H1', R['H1_verdict'], '| H2', R['H2_verdict'])
    for f in FMT: h = R['H1'][f]; print(f"  {f:6s} median {h['median']:.3f}  positive {h['frac_positive']:.2f}  cells {h['n_cells']}")
    print('  H2', R['H2_summary'])

if __name__ == '__main__': main()
