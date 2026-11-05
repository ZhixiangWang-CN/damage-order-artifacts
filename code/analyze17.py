"""Addendum 16: is the frequency effect present under each damage separately?
usage: python3 analyze17.py   # locked analysis -> analysis17.json
"""
import json, pickle
import numpy as np
from scipy.stats import spearmanr
from analyze15 import MODELS, DMG, COMPRESS, CACHE, fact_countries

def surv(C, m, meas, dmgs):
    cs = [c['res'][meas]['correct'] for k, c in C[m]['cond'].items() if k[0] in dmgs]
    return np.mean(cs, 0) if cs else None

def rho_fact(C, m, dmgs, F15, cs, countries):
    base = C[m]['cond'][('none', 0.0, 0)]; ok0 = base['res']['FACT_BARE']['correct']; s = surv(C, m, 'FACT_BARE', dmgs)
    if s is None: return None
    keep = [c for c in countries if ok0[cs == c].all()]
    sv = np.array([s[cs == c].mean() for c in keep]); lf = np.array([np.log10(F15[c]['cooc'] + 1) for c in keep])
    if len(keep) < 30 or np.ptp(sv) == 0: return None
    return float(spearmanr(lf, sv)[0])

def rho_item(C, m, task, dmgs, F16):
    ev = json.load(open(f'items_v8/{task}.json'))['eval']; lf = np.array([np.log10(F16[f"{task}|{it['ans']}"]['count'] + 1) for it in ev])
    base = C[m]['cond'][('none', 0.0, 0)]; ok0 = base['res'][f'{task}_BARE']['correct'] == 1; s = surv(C, m, f'{task}_BARE', dmgs)
    if s is None or ok0.sum() < 30 or np.ptp(s[ok0]) == 0: return None
    return float(spearmanr(lf[ok0], s[ok0])[0])

def summarize(vals):
    v = [x for x in vals if x is not None]
    return {'n_cells': len(v), 'n_missing': len(vals) - len(v), 'median': float(np.median(v)) if v else None,
            'frac_positive': float(np.mean([x > 0 for x in v])) if v else None, 'values': vals}

def main():
    C = pickle.load(open(CACHE, 'rb')); F15 = json.load(open('freq15.json'))['counts']; F16 = json.load(open('freq16.json'))['counts']
    cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); R = {'H1': {}, 'H2': {}, 'H3': {}}
    for d in DMG:
        cells = {f'{m}|FACT': rho_fact(C, m, [d], F15, cs, countries) for m in MODELS}
        cells.update({f'{m}|SCIQ': rho_item(C, m, 'SCIQ', [d], F16) for m in MODELS})
        R['H1'][d] = summarize(list(cells.values())); R['H1'][d]['cells'] = cells
        R['H1'][d]['pass'] = bool(R['H1'][d]['median'] is not None and R['H1'][d]['median'] >= 0.20 and R['H1'][d]['frac_positive'] >= 0.75)
        R['H3'][d] = summarize([rho_item(C, m, 'RETR', [d], F16) for m in MODELS])
    npass = sum(R['H1'][d]['pass'] for d in DMG); ncomp = sum(R['H1'][d]['pass'] for d in COMPRESS)
    R['H1_verdict'] = 'COMMIT' if npass == 7 else ('PARTIAL' if npass >= 5 and ncomp >= 2 else 'STOP')
    R['H1_n_pass'] = npass; R['H1_n_compression_pass'] = ncomp
    R['H2']['FACT'] = summarize([rho_fact(C, m, COMPRESS, F15, cs, countries) for m in MODELS])
    R['H2']['SCIQ'] = summarize([rho_item(C, m, 'SCIQ', COMPRESS, F16) for m in MODELS])
    json.dump(R, open('analysis17.json', 'w'), indent=1)
    print('H1', R['H1_verdict'], f'{npass}/7 damages pass, {ncomp}/3 compression')
    for d in DMG:
        h = R['H1'][d]; print(f"  {d:10s} median {h['median']:.3f}  positive {h['frac_positive']:.2f}  cells {h['n_cells']} (missing {h['n_missing']})  pass {h['pass']}  | RETR median {R['H3'][d]['median']}")
    for t in ('FACT', 'SCIQ'): print('H2 compression', t, round(R['H2'][t]['median'], 3), R['H2'][t]['values'])

if __name__ == '__main__': main()
