"""Addendum 15: frequency effect on SciQ (F1), in-context negative control RETR (F2), subject vs answer frequency in FACT (F3).
usage: python3 analyze16.py fetch   # resumable infini-gram counts -> freq16.json (no statistics)
       python3 analyze16.py run     # locked analysis -> analysis16.json
"""
import sys, os, json, time, pickle, urllib.request
import numpy as np
from scipy.stats import spearmanr
from analyze15 import MODELS, DMG, CACHE, INDEX, pspear, fact_countries

PART = 'freq16_partial.json'

def ig(query):
    req = urllib.request.Request('https://api.infini-gram.io/', data=json.dumps({'index': INDEX, 'query_type': 'count', 'query': query}).encode(),
                                 headers={'Content-Type': 'application/json'})
    for a in range(8):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=90).read())
            if 'count' in r: return r
            err = r
        except Exception as e: err = e
        time.sleep(min(20 * (a + 1), 120))
    raise RuntimeError(f'{query}: {err}')

def queries():
    Q = []
    for t in ('SCIQ', 'RETR'):
        Q += [(t, a) for a in sorted({it['ans'] for it in json.load(open(f'items_v8/{t}.json'))['eval']})]
    Q += [('COUNTRY', c) for c in sorted(set(fact_countries()[0]))]
    return Q

def fetch():
    out = json.load(open(PART)) if os.path.exists(PART) else {}
    for t, s in queries():
        k = f'{t}|{s}'
        if k in out: continue
        r = ig(s); out[k] = {'count': r['count'], 'ntok': len(r['tokens'])}
        json.dump(out, open(PART, 'w')); print(len(out), k, r['count'], flush=True); time.sleep(2)
    assert all(f'{t}|{s}' in out for t, s in queries())
    json.dump({'index': INDEX, 'counts': out}, open('freq16.json', 'w'), indent=1); print('done')

def task_test(C, F, t):
    ev = json.load(open(f'items_v8/{t}.json'))['eval']; ans = [it['ans'] for it in ev]
    lf = np.array([np.log10(F[f'{t}|{a}']['count'] + 1) for a in ans]); nt = np.array([F[f'{t}|{a}']['ntok'] for a in ans])
    per = {}
    for m in MODELS:
        cd = C[m]['cond']; base = cd[('none', 0.0, 0)]; dam = [c for k, c in cd.items() if k[0] in DMG]; row = {}
        sc = base['res'][f'{t}_CHOICE']['scores']; marg = sc[:, 0] - sc[:, 1:].max(1)
        for fmt in ('BARE', 'CHOICE'):
            meas = f'{t}_{fmt}'; ok0 = base['res'][meas]['correct'] == 1
            if ok0.sum() < 30: row[fmt] = {'n_items': int(ok0.sum())}; continue
            surv = np.mean([c['res'][meas]['correct'] for c in dam], 0)[ok0]
            row[fmt] = {'n_items': int(ok0.sum()), 'rho': float(spearmanr(lf[ok0], surv)[0]),
                        'partial_ntok': pspear(lf[ok0], surv, nt[ok0]), 'partial_margin': pspear(lf[ok0], surv, marg[ok0]),
                        'rho_margin': float(spearmanr(marg[ok0], surv)[0])}
        per[m] = row
    rho = [per[m]['BARE']['rho'] for m in MODELS if 'rho' in per[m]['BARE']]
    return per, rho

def run():
    C = pickle.load(open(CACHE, 'rb')); F = json.load(open('freq16.json'))['counts']; R = {}
    per, rho = task_test(C, F, 'SCIQ'); med, pos = float(np.median(rho)), int(sum(r > 0 for r in rho))
    R['F1_SCIQ'] = {'per_model': per, 'n_models': len(rho), 'median_rho': med, 'n_positive': pos,
                    'verdict': 'COMMIT' if med >= 0.30 and pos >= 7 else ('PARTIAL' if med >= 0.15 and pos >= 6 else 'STOP')}
    per, rho = task_test(C, F, 'RETR'); med = float(np.median(rho))
    R['F2_RETR'] = {'per_model': per, 'n_models': len(rho), 'median_rho': med, 'n_positive': int(sum(r > 0 for r in rho)),
                    'verdict': 'PASS' if med < 0.10 else ('FAIL' if med >= 0.15 else 'INCONCLUSIVE')}
    F15 = json.load(open('freq15.json'))['counts']; cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); per = {}
    for m in MODELS:
        cd = C[m]['cond']; base = cd[('none', 0.0, 0)]; dam = [c for k, c in cd.items() if k[0] in DMG]
        ok0 = base['res']['FACT_BARE']['correct']; surv = np.mean([c['res']['FACT_BARE']['correct'] for c in dam], 0)
        keep = [c for c in countries if ok0[cs == c].all()]
        s = np.array([surv[cs == c].mean() for c in keep])
        fc = np.array([np.log10(F[f'COUNTRY|{c}']['count'] + 1) for c in keep]); fa = np.array([np.log10(F15[c]['cap'] + 1) for c in keep])
        per[m] = {'n_countries': len(keep), 'rho_country': float(spearmanr(fc, s)[0]), 'rho_capital': float(spearmanr(fa, s)[0]),
                  'country_given_capital': pspear(fc, s, fa), 'capital_given_country': pspear(fa, s, fc), 'rho_country_capital': float(spearmanr(fc, fa)[0])}
    R['F3_FACT_subject_vs_answer'] = {'per_model': per}
    json.dump(R, open('analysis16.json', 'w'), indent=1)
    print('F1 SciQ', R['F1_SCIQ']['verdict'], round(R['F1_SCIQ']['median_rho'], 3), f"{R['F1_SCIQ']['n_positive']}/{R['F1_SCIQ']['n_models']}")
    print('F2 RETR', R['F2_RETR']['verdict'], round(R['F2_RETR']['median_rho'], 3), f"{R['F2_RETR']['n_positive']}/{R['F2_RETR']['n_models']}")

if __name__ == '__main__': {'fetch': fetch, 'run': run}[sys.argv[1]]()
