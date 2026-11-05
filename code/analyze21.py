"""Addendum 19: predict accuracy under compression with no damaged data from the target model.
usage: python3 analyze21.py   # locked analysis -> analysis21.json
For target model m, exponents are predicted from m's intact model only:
  log n_(t,f) = slope * z(margin_t) + beta_f, z over the 16 tasks of m (margins differ in scale between models)
  margin_t = mean over eval items of [CHOICE score of the correct candidate - best distractor] in the intact model of m;
  slope = pooled least squares of the other seven models' random-damage task effects on their own z-scored margins;
  beta_f = mean random-damage format effect of the other seven models.
Accuracy is then predicted exactly as in analyze19 (leave one measure out, one strength per compression condition), against the
uniform baseline (every exponent 1). The random-damage profile of analyze19 is reported for reference only.
"""
import json, pickle
import numpy as np
from analyze9 import additive, LO, HI
from analyze15 import MODELS, COMPRESS, CACHE
from analyze19 import cval, random_exponents, predict_condition

def margins(C, m):
    base = C[m]['cond'][('none', 0.0, 0)]['res']
    return {t[:-7]: float(np.mean([s[0] - max(s[1:]) for s in base[t]['scores']])) for t in base if t.endswith('_CHOICE')}

def main():
    C = pickle.load(open(CACHE, 'rb'))
    MG = {m: margins(C, m) for m in MODELS}; AD = {m: additive(random_exponents(C, m)) for m in MODELS}
    R = {'cells': {}, 'slopes': {}, 'H1': {}}
    for m in MODELS:
        X, Y, B = [], [], {}
        for o in MODELS:
            if o == m or AD[o] is None: continue
            ts = [t for t in AD[o]['a'] if t in MG[o]]; mc, sd = np.mean(list(MG[o].values())), np.std(list(MG[o].values()))
            X += [(MG[o][t] - mc) / sd for t in ts]; Y += [AD[o]['a'][t] for t in ts]
            for f, v in AD[o]['b'].items(): B.setdefault(f, []).append(v)
        X, Y = np.array(X), np.array(Y); slope = float((X * Y).sum() / (X * X).sum()); beta = {f: float(np.mean(v)) for f, v in B.items()}
        R['slopes'][m] = slope
        cond, base = C[m]['cond'], C[m]['cond'][('none', 0.0, 0)]
        ms = [x for x in sorted(C[m]['usable']) if base['res'][x]['acc'] - base['res'][x]['chance'] > 0 and x.rsplit('_', 1)[0] in MG[m] and x.rsplit('_', 1)[1] in beta]
        mc, sd = np.mean(list(MG[m].values())), np.std(list(MG[m].values()))
        exz = {x: float(np.exp(slope * (MG[m][x.rsplit('_', 1)[0]] - mc) / sd + beta[x.rsplit('_', 1)[1]])) for x in ms}
        for k in COMPRESS:
            rows, conds = [], 0
            for key in sorted(cond):
                if key[0] != k: continue
                if not (0.10 <= np.median([cval(base, cond[key], x) for x in exz]) <= 0.90): continue
                conds += 1; rows += predict_condition(base, cond[key], exz)
            if conds == 0 or len(rows) < 20: R['cells'][f'{m}|{k}'] = None; continue
            o, p, u = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
            R['cells'][f'{m}|{k}'] = {'n_conditions': conds, 'n_points': len(rows), 'mae_intact': float(np.abs(p - o).mean()), 'mae_uniform': float(np.abs(u - o).mean()),
                                      'r2_intact': float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())}
    ref = json.load(open('analysis19.json'))['cells']
    for k in COMPRESS:
        keys = [key for key, c in R['cells'].items() if key.endswith('|' + k) and c]; cs = [R['cells'][key] for key in keys]
        red = [1 - c['mae_intact'] / c['mae_uniform'] for c in cs]
        h = {'n_cells': len(cs), 'median_reduction': float(np.median(red)) if cs else None, 'frac_better': float(np.mean([c['mae_intact'] < c['mae_uniform'] for c in cs])) if cs else None,
             'median_mae_intact': float(np.median([c['mae_intact'] for c in cs])) if cs else None, 'median_mae_uniform': float(np.median([c['mae_uniform'] for c in cs])) if cs else None,
             'median_r2_intact': float(np.median([c['r2_intact'] for c in cs])) if cs else None,
             'reference_median_mae_random_profile': float(np.median([ref[key]['mae_profile'] for key in keys if ref.get(key)])) if cs else None}
        h['pass'] = bool(cs and len(cs) >= 4 and h['median_reduction'] >= 0.20 and h['frac_better'] >= 0.75); R['H1'][k] = h
    n1 = sum(R['H1'][k]['pass'] for k in COMPRESS); R['H1_verdict'] = 'COMMIT' if n1 == 3 else ('PARTIAL' if n1 == 2 else 'STOP')
    json.dump(R, open('analysis21.json', 'w'), indent=1)
    print('H1', R['H1_verdict'], f'{n1}/3')
    for k in COMPRESS:
        h = R['H1'][k]
        print(f"  {k:10s} cells {h['n_cells']}  MAE intact-only {h['median_mae_intact']:.3f} vs uniform {h['median_mae_uniform']:.3f}  reduction {h['median_reduction']:.2f}  "
              f"better {h['frac_better']:.2f}  R2 {h['median_r2_intact']:.2f}  | random-damage profile {h['reference_median_mae_random_profile']:.3f}")

if __name__ == '__main__': main()
