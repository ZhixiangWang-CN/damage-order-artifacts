"""Exploratory (not preregistered): the three axes on one scale.
Survival of a measure = mean over all damaged conditions (7 damages, all doses and seeds) of the chance-corrected score relative to the
intact model, c = (acc - chance) / (acc0 - chance), clipped to [0, 1]. For facts, survival of a unit is its mean correctness over the same
conditions among units the intact model answered correctly (which is the same c for one unit).
  format : survival(BARE) - survival(TWICE) and survival(CHOICE) - survival(TWICE), per task, median over tasks
  task   : |survival(t1) - survival(t2)| over task pairs within one format (typical pair = median), and most - least robust task
  fact   : common half - rare half (median split of log answer frequency), and top - bottom quartile; capitals and SciQ, BARE
Medians over models are reported. usage: python3 analyze23_explore.py -> analysis23_explore.json
"""
import json, pickle, itertools
import numpy as np
from analyze15 import MODELS, DMG, CACHE, fact_countries
from analyze22 import unit_table

FMT4 = ['CHOICE', 'BARE', 'JSON', 'TWICE']

def survival(C, m):
    cond = C[m]['cond']; base = cond[('none', 0.0, 0)]['res']; out = {}
    for x in C[m]['usable']:
        a0, k = base[x]['acc'], base[x]['chance']
        if a0 - k <= 0.05: continue
        out[x] = float(np.mean([np.clip((cond[key]['res'][x]['acc'] - k) / (a0 - k), 0, 1) for key in cond if key[0] in DMG]))
    return out

def main():
    C = pickle.load(open(CACHE, 'rb')); F15 = json.load(open('freq15.json'))['counts']; F16 = json.load(open('freq16.json'))['counts']
    cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); R = {}
    for m in MODELS:
        S = survival(C, m); tasks = sorted({x.rsplit('_', 1)[0] for x in S})
        bt = [S[f'{t}_BARE'] - S[f'{t}_TWICE'] for t in tasks if f'{t}_BARE' in S and f'{t}_TWICE' in S]
        ct = [S[f'{t}_CHOICE'] - S[f'{t}_TWICE'] for t in tasks if f'{t}_CHOICE' in S and f'{t}_TWICE' in S]
        pair, span = [], []
        for f in FMT4:
            v = [S[f'{t}_{f}'] for t in tasks if f'{t}_{f}' in S]
            if len(v) >= 8: pair.append(np.median([abs(a - b) for a, b in itertools.combinations(v, 2)])); span.append(max(v) - min(v))
        fh, fq = [], []
        for t in ('FACT', 'SCIQ'):
            lf, T = unit_table(C, m, t, F15, F16, cs, countries)
            if 'BARE' not in T: continue
            ok, sv = T['BARE']; l, s = lf[ok], sv[ok]
            if ok.sum() < 30: continue
            md = np.median(l); q1, q3 = np.percentile(l, [25, 75])
            fh.append(s[l >= md].mean() - s[l < md].mean()); fq.append(s[l >= q3].mean() - s[l <= q1].mean())
        R[m] = {'format_bare_minus_twice': float(np.median(bt)), 'format_choice_minus_twice': float(np.median(ct)),
                'task_typical_pair': float(np.median(pair)), 'task_most_minus_least': float(np.median(span)),
                'fact_common_minus_rare_half': float(np.mean(fh)) if fh else None, 'fact_top_minus_bottom_quartile': float(np.mean(fq)) if fq else None}
    keys = list(next(iter(R.values())))
    summ = {k: {'median': float(np.median([R[m][k] for m in R if R[m][k] is not None])), 'range': [float(min(R[m][k] for m in R if R[m][k] is not None)), float(max(R[m][k] for m in R if R[m][k] is not None))]} for k in keys}
    json.dump({'per_model': R, 'summary': summ}, open('analysis23_explore.json', 'w'), indent=1)
    for k, v in summ.items(): print(f"{k:32s} median {v['median']:.3f}  range {v['range'][0]:.3f}-{v['range'][1]:.3f}")

if __name__ == '__main__': main()
