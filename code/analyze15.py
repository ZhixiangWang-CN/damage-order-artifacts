"""Addendum 14: template tripwire (D1), frequency effect (D2a), item-level shared order (D2b).
usage: python3 analyze15.py cache   # extract compact per-item arrays from runs_v8/runs_v9 (no statistics)
       python3 analyze15.py freq    # infini-gram counts for the 78 FACT countries -> freq15.json
       python3 analyze15.py run     # locked analysis -> analysis15.json
"""
import sys, os, json, gzip, glob, re, pickle, time, itertools, urllib.request
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.isotonic import IsotonicRegression

MODELS = ['q15', 'llama', 'phi', 'q7', 'mis', 'gemma', 'q14', 'q32']
DMG = ['noise', 'zero', 'neuron', 'ld_rand', 'wanda', 'sparsegpt', 'gptq']
RANDOM = ['noise', 'zero', 'neuron', 'ld_rand']
COMPRESS = ['wanda', 'sparsegpt', 'gptq']
CACHE = os.environ.get('A15_CACHE', 'analysis15_cache.pkl')
INDEX = 'v4_dolma-v1_7_llama'

def files_for(m):
    out = {}
    for d in [f'runs_v9/all/{m}', f'runs_v8/server_a/{m}', f'runs_v8/server_b/{m}']:
        for p in sorted(glob.glob(f'{d}/*.json.gz')):
            dmg, dose, seed = os.path.basename(p)[:-8].rsplit('_', 2)
            if dmg in DMG or dmg == 'none': out.setdefault((dmg, float(dose), int(seed)), p)
    return out

def cache():
    C = {}
    for m in MODELS:
        C[m] = {'usable': sorted(json.load(open(f'runs_v9/all/{m}/usable.json'))['usable']), 'cond': {}}
        fs = files_for(m)
        assert ('none', 0.0, 0) in fs, m
        for k, p in fs.items():
            d = json.load(gzip.open(p, 'rt'))
            C[m]['cond'][k] = {'nll': d.get('probe', {}).get('nll'), 'kl': d.get('probe', {}).get('kl'),
                               'res': {meas: {'acc': v['acc'], 'chance': v['chance'], 'correct': np.array(v['correct'], np.int8),
                                              'format_ok': np.array(v['format_ok'], np.int8) if 'format_ok' in v else None,
                                              'scores': np.array(v['scores'], np.float32) if 'scores' in v else None}
                                       for meas, v in d['res'].items() if meas != 'GSM8K_COT'}}
        print(m, len(C[m]['cond']), 'conditions', flush=True)
    pickle.dump(C, open(CACHE, 'wb'))

# ---------------------------------------------------------------- frequencies (D2a)
def fact_countries():
    ev = json.load(open('items_v8/FACT.json'))['eval']
    return [re.match(r'What is the capital of (.+)\?$', it['q']).group(1) for it in ev], [it['ans'] for it in ev]

def ig(query):
    req = urllib.request.Request('https://api.infini-gram.io/', data=json.dumps({'index': INDEX, 'query_type': 'count', 'query': query}).encode(),
                                 headers={'Content-Type': 'application/json'})
    for a in range(5):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            if 'count' in r: return r
        except Exception as e: err = e
        time.sleep(2 + 3 * a)
    raise RuntimeError(f'{query}: {err}')

def freq():
    cs, ans = fact_countries(); pairs = sorted(set(zip(cs, ans))); out = {}
    for c, a in pairs:
        co, cap, ph = ig(f'{c} AND {a}'), ig(a), ig(f'capital of {c} is {a}')
        out[c] = {'capital': a, 'cooc': co['count'], 'cooc_approx': co.get('approx'), 'cap': cap['count'], 'cap_ntok': len(cap['tokens']), 'phrase': ph['count']}
        print(c, a, out[c]['cooc'], out[c]['cap'], out[c]['phrase'], flush=True); time.sleep(0.3)
    json.dump({'index': INDEX, 'counts': out}, open('freq15.json', 'w'), indent=1)

# ---------------------------------------------------------------- helpers
def cc(acc, ch): return (acc - ch) / (1 - ch)
def r2(y, yh): y, yh = np.asarray(y), np.asarray(yh); return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def pspear(x, y, z):
    """partial Spearman of x and y controlling z (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])

# ---------------------------------------------------------------- D1
def d1_points(C):
    P = []
    for m in MODELS:
        cd = C[m]['cond']; base = cd[('none', 0.0, 0)]
        cb = [u for u in C[m]['usable'] if u.endswith('_CHOICE') or u.endswith('_BARE')]
        tw = sorted(k for k in base['res'] if k.endswith('_TWICE'))
        A = lambda c: np.mean([cc(c['res'][u]['acc'], c['res'][u]['chance']) for u in cb])
        B = lambda c: np.mean([1 - c['res'][k]['format_ok'].mean() for k in tw])
        A0, B0, n0 = A(base), B(base), base['nll']
        for k, c in cd.items():
            if k[0] == 'none': continue
            P.append({'model': m, 'damage': k[0], 'dose': k[1], 'seed': k[2], 'L': float(1 - A(c) / A0),
                      'S_T': float((B(c) - B0) / (1 - B0)), 'S_ppl': float((c['nll'] - n0) / n0), 'S_KL': float(c['kl'])})
    return P

def lomo(P, sig, keep=lambda p: True):
    y, yh = [], []
    for m in MODELS:
        tr = [p for p in P if p['model'] != m]; te = [p for p in P if p['model'] == m and p['L'] <= 0.5 and keep(p)]
        if not te: continue
        f = IsotonicRegression(increasing=True, out_of_bounds='clip').fit([p[sig] for p in tr], [p['L'] for p in tr])
        y += [p['L'] for p in te]; yh += list(f.predict([p[sig] for p in te]))
    return r2(y, yh), len(y)

def d1(C):
    P = d1_points(C); S = ['S_T', 'S_ppl', 'S_KL']; out = {'n_points': len(P)}
    out['R2'] = {s: lomo(P, s)[0] for s in S}; out['n_eval'] = lomo(P, 'S_T')[1]
    out['R2_compression'] = {s: lomo(P, s, lambda p: p['damage'] in COMPRESS)[0] for s in S}
    out['R2_random'] = {s: lomo(P, s, lambda p: p['damage'] in RANDOM)[0] for s in S}
    out['R2_by_damage'] = {d: {s: lomo(P, s, lambda p, d=d: p['damage'] == d)[0] for s in S} for d in DMG}
    early = [p for p in P if p['L'] <= 0.15]
    out['early_spearman'] = {s: float(spearmanr([p[s] for p in early], [p['L'] for p in early])[0]) for s in S}; out['n_early'] = len(early)
    rT, best = out['R2']['S_T'], max(out['R2']['S_ppl'], out['R2']['S_KL'])
    out['verdict'] = 'COMMIT' if rT >= 0.5 and rT - best >= 0.10 else ('TIE' if rT >= 0.5 else 'STOP')
    out['points'] = P
    return out

# ---------------------------------------------------------------- D2a
def d2a(C):
    F = json.load(open('freq15.json'))['counts']; cs, _ = fact_countries(); cs = np.array(cs); countries = sorted(set(cs))
    out = {'per_model': {}}
    for m in MODELS:
        cd = C[m]['cond']; base = cd[('none', 0.0, 0)]; dam = [c for k, c in cd.items() if k[0] in DMG]
        sc = base['res']['FACT_CHOICE']['scores']; marg = sc[:, 0] - sc[:, 1:].max(1)
        row = {}
        for fmt in ['BARE', 'CHOICE']:
            meas = f'FACT_{fmt}'; ok0 = base['res'][meas]['correct']
            surv = np.mean([c['res'][meas]['correct'] for c in dam], 0)
            keep = [c for c in countries if ok0[cs == c].all()]
            s = np.array([surv[cs == c].mean() for c in keep]); lf = np.array([np.log10(F[c]['cooc'] + 1) for c in keep])
            r = {'n_countries': len(keep), 'rho_cooc': float(spearmanr(lf, s)[0]),
                 'rho_cap': float(spearmanr([np.log10(F[c]['cap'] + 1) for c in keep], s)[0]),
                 'rho_phrase': float(spearmanr([np.log10(F[c]['phrase'] + 1) for c in keep], s)[0]),
                 'partial_ntok': pspear(lf, s, [F[c]['cap_ntok'] for c in keep]),
                 'partial_margin': pspear(lf, s, [marg[cs == c].mean() for c in keep]),
                 'rho_margin': float(spearmanr([marg[cs == c].mean() for c in keep], s)[0])}
            row[fmt] = r
        out['per_model'][m] = row
    rho = [out['per_model'][m]['BARE']['rho_cooc'] for m in MODELS]
    med, pos = float(np.median(rho)), int(sum(r > 0 for r in rho))
    out.update({'median_rho': med, 'n_positive': pos,
                'verdict': 'COMMIT' if med >= 0.30 and pos >= 7 else ('PARTIAL' if med >= 0.15 and pos >= 6 else 'STOP')})
    return out

# ---------------------------------------------------------------- D2b
def d2b(C):
    out = {'per_model': {}}
    for m in MODELS:
        cd = C[m]['cond']; base = cd[('none', 0.0, 0)]; cross, within = [], []; nm = 0
        for meas in C[m]['usable']:
            ok0 = base['res'][meas]['correct'] == 1
            if ok0.sum() < 30: continue
            surv = {}
            for d in RANDOM:
                for s in (1, 2):
                    cs = [c['res'][meas]['correct'][ok0] for k, c in cd.items() if k[0] == d and k[2] == s]
                    if cs: surv[(d, s)] = np.mean(cs, 0)
            if any(np.ptp(v) == 0 for v in surv.values()): continue
            ds = [d for d in RANDOM if (d, 1) in surv and (d, 2) in surv]
            if len(ds) < 2: continue
            w = [spearmanr(surv[(d, 1)], surv[(d, 2)])[0] for d in ds]
            x = [(spearmanr(surv[(a, 1)], surv[(b, 2)])[0] + spearmanr(surv[(a, 2)], surv[(b, 1)])[0]) / 2 for a, b in itertools.combinations(ds, 2)]
            within.append(np.median(w)); cross.append(np.median(x)); nm += 1
        rc, rw = float(np.median(cross)), float(np.median(within))
        out['per_model'][m] = {'n_measures': nm, 'r_cross': rc, 'r_within': rw, 'ratio': rc / rw}
    ratio = [out['per_model'][m]['ratio'] for m in MODELS]; rc = [out['per_model'][m]['r_cross'] for m in MODELS]
    out.update({'median_ratio': float(np.median(ratio)), 'median_r_cross': float(np.median(rc))})
    out['verdict'] = 'COMMIT' if out['median_ratio'] >= 0.80 and out['median_r_cross'] >= 0.30 else ('PARTIAL' if out['median_ratio'] >= 0.60 else 'STOP')
    return out

def run():
    C = pickle.load(open(CACHE, 'rb'))
    R = {'D1': d1(C), 'D2a': d2a(C), 'D2b': d2b(C)}
    json.dump(R, open('analysis15.json', 'w'), indent=1)
    print('D1', R['D1']['verdict'], {k: round(v, 3) for k, v in R['D1']['R2'].items()})
    print('D2a', R['D2a']['verdict'], round(R['D2a']['median_rho'], 3), f"{R['D2a']['n_positive']}/8")
    print('D2b', R['D2b']['verdict'], round(R['D2b']['median_ratio'], 3), round(R['D2b']['median_r_cross'], 3))

if __name__ == '__main__': {'cache': cache, 'freq': freq, 'run': run}[sys.argv[1]]()
