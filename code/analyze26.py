"""Addendum 22: do the principal conclusions survive separate scoring of content, template and strict success?
usage: python3 analyze26.py cache   # per-measure accuracies under each scoring from the raw runs -> analysis26_cache.pkl (no statistics)
       python3 analyze26.py smoke   # strict-scoring reproduction only (values already in the paper), no rescored statistic
       python3 analyze26.py run     # locked analysis -> analysis26.json

Scorings of the generated formats (BARE, JSON, TWICE); CHOICE is a ranking of candidates and is the same in all of them.
  strict    correct = template valid and answer correct (the paper's scoring), chance 1/k for listed options and yes/no, else 0
  strict0   the same correctness with chance 0 for every generated measure (a model that writes invalid templates does not
            reach 1/k by guessing)
  content   the scorer's own content_any field, written with the run: the gold answer is present in the output whatever the
            template (first standalone letter or yes/no for option and yes/no tasks; any matching number or name otherwise);
            chance as in strict
  template  format_ok, the template alone, chance 0 (report only)
Principal conclusions re-tested under content and strict0:
  C1 shared order: within each model, Pearson correlation of task effects between pairs of the seven damages (>= 8 common tasks)
  C2 writing fails before choosing: format effect of CHOICE below that of BARE, averaged over the random damages
  C3 order under compression: Pearson of compression log exponents with the random-damage reference (log mean, >= 2 damages)
  C4 accuracy: random-damage profile against uniform decline (analyze19 predictor, leave one measure out)
"""
import sys, json, gzip, pickle, itertools
import numpy as np
from analyze9 import rank1, additive, LO, HI, MIN_MEASURES, MIN_COMMON_T
from analyze15 import MODELS, DMG, RANDOM, COMPRESS, files_for
from analyze19 import cval, random_exponents, predict_condition

CACHE26 = 'analysis26_cache.pkl'; GEN = ('BARE', 'JSON', 'TWICE'); SCORINGS = ('strict', 'strict0', 'content', 'template')

def cache():
    out = {}
    for m in MODELS:
        out[m] = {'usable': sorted(json.load(open(f'runs_v9/all/{m}/usable.json'))['usable']), 'cond': {}}
        for key, p in files_for(m).items():
            d = json.load(gzip.open(p, 'rt'))['res']; r = {}
            for meas, v in d.items():
                if meas == 'GSM8K_COT': continue
                r[meas] = {'correct': np.array(v['correct'], np.int8), 'chance': v['chance'],
                           'format_ok': np.array(v['format_ok'], np.int8) if 'format_ok' in v else None,
                           'content': np.array(v['content_any'], np.int8) if 'content_any' in v else None}
            out[m]['cond'][key] = r
        print(m, len(out[m]['cond']), flush=True)
    pickle.dump(out, open(CACHE26, 'wb'))

def view(R, scoring):
    """a cache in the analysis15 layout (acc, chance per measure) for one scoring"""
    C = {}
    for m, x in R.items():
        C[m] = {'usable': x['usable'], 'cond': {}}
        for key, res in x['cond'].items():
            rr = {}
            for meas, v in res.items():
                gen = meas.rsplit('_', 1)[1] in GEN
                if not gen or scoring == 'strict': a, k = v['correct'].mean(), v['chance']
                elif scoring == 'strict0': a, k = v['correct'].mean(), 0.0
                elif scoring == 'content': a, k = v['content'].mean(), v['chance']
                else: a, k = v['format_ok'].mean(), 0.0
                rr[meas] = {'acc': float(a), 'chance': float(k)}
            C[m]['cond'][key] = {'res': rr}
    return C

def usable(C, m):
    b = C[m]['cond'][('none', 0.0, 0)]['res']
    return [x for x in sorted(C[m]['usable']) if x in b and b[x]['acc'] - b[x]['chance'] > 0.05]

def cell_exponents(C, m, k):
    cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]; ms = usable(C, m)
    doses = sorted({d for (kk, d, s) in cond if kk == k})
    if len(doses) < 3: return None
    Cm = np.array([[np.mean([cval(base, cond[(k, d, s)], x) for (kk, dd, s) in cond if kk == k and dd == d]) for d in doses] for x in ms])
    M = ((Cm > LO) & (Cm < HI)).astype(float); keep = M.sum(1) >= 2
    if keep.sum() < MIN_MEASURES: return None
    n, _ = rank1(-np.log(np.clip(Cm[keep], 1e-6, 1)), M[keep])
    return dict(zip([x for x, q in zip(ms, keep) if q], n.tolist()))

def pear_common(x, y, mn):
    if not x or not y: return None
    ks = [k for k in x if k in y]
    return float(np.corrcoef([x[k] for k in ks], [y[k] for k in ks])[0, 1]) if len(ks) >= mn else None

def statistics(C):
    S = {'C1': {}, 'C2': {}, 'C3': {}, 'C4': {}}
    for m in MODELS:
        E = {k: cell_exponents(C, m, k) for k in DMG}; A = {k: additive(E[k]) for k in DMG}
        S['C1'][m] = [r for a, b in itertools.combinations(DMG, 2)
                      if (r := pear_common((A[a] or {}).get('a'), (A[b] or {}).get('a'), MIN_COMMON_T)) is not None]
        bs = [A[k]['b'] for k in RANDOM if A[k] and 'CHOICE' in A[k]['b'] and 'BARE' in A[k]['b']]
        S['C2'][m] = {f: float(np.mean([b[f] for b in bs if f in b])) for f in ('CHOICE', 'BARE', 'JSON', 'TWICE') if any(f in b for b in bs)} if bs else None
        logs = {}
        for k in RANDOM:
            for x, v in (E[k] or {}).items(): logs.setdefault(x, []).append(np.log(v))
        ref = {x: float(np.mean(v)) for x, v in logs.items() if len(v) >= 2}
        S['C3'][m] = {k: pear_common(ref, {x: float(np.log(v)) for x, v in E[k].items()} if E[k] else None, 10) for k in COMPRESS}
        ex = random_exponents(C, m); cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]; S['C4'][m] = {}
        for k in COMPRESS:
            rows = []
            for key in sorted(cond):
                if key[0] != k or not ex or not (0.10 <= np.median([cval(base, cond[key], x) for x in ex]) <= 0.90): continue
                rows += predict_condition(base, cond[key], ex)
            if len(rows) < 20: S['C4'][m][k] = None; continue
            o, p, u = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
            S['C4'][m][k] = {'mae_profile': float(np.abs(p - o).mean()), 'mae_uniform': float(np.abs(u - o).mean())}
    allc1 = [r for v in S['C1'].values() for r in v]
    c2 = [v for v in S['C2'].values() if v]
    c3 = [r for v in S['C3'].values() for r in v.values() if r is not None]
    c4 = {k: [v[k] for v in S['C4'].values() if v.get(k)] for k in COMPRESS}
    summ = {'C1_median': float(np.median(allc1)) if allc1 else None, 'C1_pairs': len(allc1),
            'C2_models_choice_below_bare': int(sum(v['CHOICE'] < v['BARE'] for v in c2)), 'C2_models': len(c2),
            'C2_mean_effects': {f: float(np.mean([v[f] for v in c2 if f in v])) for f in ('CHOICE', 'BARE', 'JSON', 'TWICE')},
            'C3_median': float(np.median(c3)) if c3 else None, 'C3_cells': len(c3),
            'C4': {k: {'n_cells': len(c), 'median_reduction': float(np.median([1 - x['mae_profile'] / x['mae_uniform'] for x in c])) if c else None,
                       'frac_better': float(np.mean([x['mae_profile'] < x['mae_uniform'] for x in c])) if c else None,
                       'median_mae_profile': float(np.median([x['mae_profile'] for x in c])) if c else None} for k, c in c4.items()}}
    return S, summ

def decomposition(R):
    """report only: among items the intact model got right (strict), the share of damaged failures that are template failures,
    per generated format, over informative random and compression conditions (median c over measures 0.10-0.90 under strict)."""
    out = {}
    Cs = view(R, 'strict')
    for m in MODELS:
        base = R[m]['cond'][('none', 0.0, 0)]; ms = usable(Cs, m); out[m] = {}
        for f in GEN:
            tf = cf = 0; fok = []
            for key, res in R[m]['cond'].items():
                if key[0] == 'none': continue
                if not (0.10 <= np.median([cval(Cs[m]['cond'][('none', 0.0, 0)], Cs[m]['cond'][key], x) for x in ms]) <= 0.90): continue
                for meas in ms:
                    if not meas.endswith('_' + f): continue
                    ok0 = base[meas]['correct'] == 1; v = res[meas]; fail = ok0 & (v['correct'] == 0)
                    tf += int((fail & (v['format_ok'] == 0)).sum()); cf += int((fail & (v['format_ok'] == 1)).sum()); fok.append(float(v['format_ok'].mean()))
            out[m][f] = {'template_share_of_failures': tf / (tf + cf) if tf + cf else None, 'mean_template_validity': float(np.mean(fok)) if fok else None}
    return out

def tests(summ):
    t = {'C1': bool(summ['C1_median'] is not None and summ['C1_median'] >= 0.70),
         'C2': bool(summ['C2_models'] >= 7 and summ['C2_models_choice_below_bare'] >= 7),
         'C3': bool(summ['C3_median'] is not None and summ['C3_median'] >= 0.70),
         'C4': bool(sum(summ['C4'][k]['n_cells'] >= 4 and summ['C4'][k]['median_reduction'] >= 0.25 and summ['C4'][k]['frac_better'] >= 0.75 for k in COMPRESS) == 3)}
    return t

def smoke():
    R = pickle.load(open(CACHE26, 'rb')); _, s = statistics(view(R, 'strict'))
    print(json.dumps({k: v for k, v in s.items()}, indent=1)[:1500])

def run():
    R = pickle.load(open(CACHE26, 'rb')); out = {'scorings': {}}
    for sc in SCORINGS:
        S, summ = statistics(view(R, sc)); out['scorings'][sc] = {'summary': summ, 'detail': S}
        if sc in ('content', 'strict0'): out['scorings'][sc]['tests'] = tests(summ)
    out['decomposition'] = decomposition(R)
    passed = {sc: all(out['scorings'][sc]['tests'].values()) for sc in ('content', 'strict0')}
    out['verdict'] = 'COMMIT' if all(passed.values()) else ('PARTIAL' if any(passed.values()) else 'STOP')
    json.dump(out, open('analysis26.json', 'w'), indent=1, default=float)
    for sc in SCORINGS:
        s = out['scorings'][sc]['summary']
        print(f"{sc:9s} C1 {s['C1_median']}  C2 {s['C2_models_choice_below_bare']}/{s['C2_models']} {s['C2_mean_effects']}  C3 {s['C3_median']}  "
              f"C4 " + ' '.join(f"{k}:{s['C4'][k]['median_reduction']}/{s['C4'][k]['frac_better']}" for k in COMPRESS), out['scorings'][sc].get('tests', ''))
    print('verdict', out['verdict'])

if __name__ == '__main__':
    {'cache': cache, 'smoke': smoke, 'run': run}[sys.argv[1]]()
