"""Locked analysis for PROTOCOL_ADDENDUM_10 (margin mechanism). Written before any margin was collected.
  python3 analyze11.py <margins_dir> <analysis10.json> <v8_root> [<v8_root2> ...]   -> analysis11.json + printed verdicts
Observed exponents come from analysis10.json (analyze9.fit_cell, unchanged); usable measures from each model's usable.json.
"""
import sys, os, gzip, json
import numpy as np
from scipy.stats import norm
import analyze9 as Z

MODELS7 = Z.MODELS; UNT = Z.UNT; COMP = ['wanda', 'sparsegpt', 'gptq']; TASKS, FORMATS = Z.TASKS, Z.FORMATS
SIG = np.logspace(-2, 2, 60); EXTREME = ['COPY', 'RETR', 'FACT', 'ARITH3']; MIN_T = 6

def pear(a, b): return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0 else None

def survival(rec, fmt, mode, const=None):
    """per intact-correct item, survival curve over SIG. mode: 'main' | 'all' | 'top2'. const replaces every gap (length-only control)."""
    S = []
    if fmt == 'CHOICE':
        for ok, sc, ln in zip(rec['correct'], rec['scores'], rec['lens']):
            if not ok: continue
            s = np.ones_like(SIG)
            for k in range(1, len(sc)):
                d = sc[0] - sc[k] if const is None else const
                s *= norm.cdf(d / (SIG * np.sqrt((ln[0] + ln[k]) / 2)))
                if mode == 'top2': break   # only the best distractor
            S.append(s)
        return S
    for ok, gp, cr in zip(rec['correct'], rec['gaps'], rec['crit']):
        if not ok: continue
        steps = gp if mode == 'all' else gp[:cr]; s = np.ones_like(SIG)
        for g in steps:
            for d in (g[:1] if mode == 'top2' else g):
                s *= norm.cdf((d if const is None else const) / SIG)
        S.append(s)
    return S

def predict(mg, ms, mode='main', const=None):
    """predicted normalized exponents for the measures ms of one model (same rank-one rule as analyze9.fit_cell)."""
    C, keep_m = [], []
    for m in ms:
        t, f = m.rsplit('_', 1); S = survival(mg[m], f, mode, const)
        if S: C.append(np.mean(S, 0)); keep_m.append(m)
    C = np.array(C); M = ((C > Z.LO) & (C < Z.HI)).astype(float); keep = M.sum(1) >= 2
    n, _ = Z.rank1(-np.log(np.clip(C[keep], 1e-6, 1)), M[keep])
    return {m: float(v) for m, v in zip([m for m, q in zip(keep_m, keep) if q], n)}

def observed(ex, m, damages):
    """measure -> mean log exponent over the damages that fitted it (at least two for the untargeted reference)."""
    vals = {}
    for k in damages:
        for mm, v in (ex.get(m, {}).get(k) or {}).items(): vals.setdefault(mm, []).append(np.log(v))
    return {mm: float(np.mean(v)) for mm, v in vals.items() if len(v) >= (2 if len(damages) > 1 else 1)}

def all_gaps(mg):
    g = [d for m, r in mg.items() if not m.endswith('_CHOICE') for ok, gp, cr in zip(r['correct'], r['gaps'], r['crit']) if ok for st in gp[:cr] for d in st]
    g += [sc[0] - sc[k] for m, r in mg.items() if m.endswith('_CHOICE') for ok, sc in zip(r['correct'], r['scores']) if ok for k in range(1, len(sc))]
    return float(np.median(g))

def tests(pred, obs, mg, drop=()):
    """M1, M2 (with baseline B2) and M3 for one model."""
    ms = [m for m in pred if m in obs and m.rsplit('_', 1)[0] not in drop]
    m1 = pear([np.log(pred[m]) for m in ms], [obs[m] for m in ms])
    cmarg = {t: np.mean([sc[0] - max(sc[1:]) for ok, sc in zip(mg[f'{t}_CHOICE']['correct'], mg[f'{t}_CHOICE']['scores'])]) for t in TASKS}
    m2, b2 = {}, {}
    for f in FORMATS:
        mf = [m for m in ms if m.endswith('_' + f)]
        if len(mf) >= MIN_T:
            m2[f] = pear([np.log(pred[m]) for m in mf], [obs[m] for m in mf]); b2[f] = pear([-cmarg[m.rsplit('_', 1)[0]] for m in mf], [obs[m] for m in mf])
    ap = Z.additive({m: pred[m] for m in ms}); ao = Z.additive({m: float(np.exp(obs[m])) for m in ms})
    m3 = None
    if ap and ao and all(f in ap['b'] for f in FORMATS) and all(f in ao['b'] for f in FORMATS):
        m3 = {'pred_b': {f: round(ap['b'][f], 3) for f in FORMATS}, 'obs_b': {f: round(ao['b'][f], 3) for f in FORMATS},
              'pred_order_exact': bool(ap['b']['CHOICE'] < ap['b']['BARE'] < ap['b']['JSON'] < ap['b']['TWICE']),
              'twice_minus_choice': {'pred': round(ap['b']['TWICE'] - ap['b']['CHOICE'], 3), 'obs': round(ao['b']['TWICE'] - ao['b']['CHOICE'], 3)}}
    return {'n_measures': len(ms), 'M1': m1, 'M2': m2, 'B2': b2, 'M3': m3}

def summarize(per):
    m1 = [v['M1'] for v in per.values() if v['M1'] is not None]
    m2 = [x for v in per.values() for x in v['M2'].values() if x is not None]; b2 = [x for v in per.values() for x in v['B2'].values() if x is not None]
    m3 = [v['M3']['pred_order_exact'] for v in per.values() if v['M3']]
    return {'M1_median': Z.med(m1), 'M2_median': Z.med(m2), 'B2_median': Z.med(b2), 'M3_exact_models': int(sum(m3)), 'M3_models': len(m3)}

def verdict(s, s_len):
    if s['M1_median'] is None: return 'not measurable'
    if s['M1_median'] < 0.65 or s['M3_exact_models'] < 4: v = 'STOP: margin mechanism stays out of the main line (appendix, negative result)'
    elif s['M1_median'] >= 0.80 and s['M3_exact_models'] >= 6 and s['M2_median'] >= s['B2_median']: v = 'COMMIT: margin mechanism becomes the main line'
    else: v = 'PARTIAL: format effect explained by the number and margin of required decisions; task effect by the answer margin'
    if s_len['M1_median'] is not None and s['M1_median'] - s_len['M1_median'] < 0.03: v += ' | wording: "number of required decisions", not "margin"'
    return v

def main(mdir, a10, v8roots):
    ex = json.load(open(a10))['exponents']; out = {'per_model': {}, 'length_only': {}, 'sensitivity': {}, 'secondary': {}}
    for m in MODELS7 + ['q32']:
        f = f'{mdir}/{m}.json.gz'
        if not os.path.exists(f): continue
        with gzip.open(f, 'rt') as fh: mg = json.load(fh)['res']
        use = next((json.load(open(f'{r}/{m}/usable.json')) for r in v8roots if os.path.exists(f'{r}/{m}/usable.json')), None)
        ms = sorted(x for x in use['usable'] if x in mg)
        pred = predict(mg, ms); obs = observed(ex, m, UNT)
        if m in MODELS7:
            out['per_model'][m] = tests(pred, obs, mg)
            out['length_only'][m] = tests(predict(mg, ms, const=all_gaps(mg)), obs, mg)
            out['sensitivity'].setdefault('all_steps', {})[m] = tests(predict(mg, ms, 'all'), obs, mg)
            out['sensitivity'].setdefault('top2_only', {})[m] = tests(predict(mg, ms, 'top2'), obs, mg)
            out['sensitivity'].setdefault('without_extreme_tasks', {})[m] = tests(pred, obs, mg, drop=EXTREME)
            out['same_as_v8'] = out.get('same_as_v8', {}); out['same_as_v8'][m] = round(float(np.mean([x for k, r in mg.items() if 'same_as_v8' in r for x in r['same_as_v8']])), 4)
        else:
            out['secondary'][f'{m}/untargeted'] = tests(pred, obs, mg) if obs else None
        for k in COMP:
            o = observed(ex, m, [k])
            if o: out['secondary'][f'{m}/{k}'] = {kk: vv for kk, vv in tests(pred, o, mg).items() if kk in ('n_measures', 'M1')}
    s = summarize(out['per_model']); sl = summarize(out['length_only'])
    out['summary'] = s; out['summary_length_only'] = sl; out['summary_sensitivity'] = {k: summarize(v) for k, v in out['sensitivity'].items()}
    out['verdict'] = verdict(s, sl)
    json.dump(out, open('analysis11.json', 'w'), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('summary', 'summary_length_only', 'summary_sensitivity', 'verdict')}, indent=1))

if __name__ == '__main__': main(sys.argv[1], sys.argv[2], sys.argv[3:])
