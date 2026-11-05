"""Exploratory: the three axes compared on one scale with one rule: top quarter minus bottom quarter of each axis.
Scale: share of ability kept over all damaged conditions (analyze23_explore.survival; facts: unit survival among intact-correct units).
  format: CHOICE minus TWICE (each format is one quarter of a four-level axis), mean over tasks (a difference of means, as for task and fact)
  task  : within each format, tasks split into quarters by an ordering fixed in advance (intact margin, analyze21.margins; 4 tasks each),
          top-margin quarter minus bottom-margin quarter; median over formats. Reported also with tasks ordered by their own survival
          (in-sample, so inflated by selection).
  fact  : most common quarter minus rarest quarter of answer frequency (BARE; capitals and SciQ, mean of the two)
Per model, then median (range) over the eight models. One step = the same difference divided by 3 (four levels -> three steps).
usage: python3 analyze24_explore.py -> analysis24_explore.json"""
import json, pickle
import numpy as np
from analyze15 import MODELS, CACHE, fact_countries
from analyze22 import unit_table
from analyze21 import margins
from analyze23_explore import survival

FMT4 = ['CHOICE', 'BARE', 'JSON', 'TWICE']
def main():
    C = pickle.load(open(CACHE, 'rb')); F15 = json.load(open('freq15.json'))['counts']; F16 = json.load(open('freq16.json'))['counts']
    cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); R = {}
    for m in MODELS:
        S = survival(C, m); MG = margins(C, m); tasks = sorted({x.rsplit('_', 1)[0] for x in S})
        fmt = [S[f'{t}_CHOICE'] - S[f'{t}_TWICE'] for t in tasks if f'{t}_CHOICE' in S and f'{t}_TWICE' in S]
        tm, ts = [], []
        for f in FMT4:
            tt = [t for t in tasks if f'{t}_{f}' in S]
            if len(tt) < 8: continue
            q = len(tt) // 4
            byM = sorted(tt, key=lambda t: -MG[t]); tm.append(np.mean([S[f'{t}_{f}'] for t in byM[:q]]) - np.mean([S[f'{t}_{f}'] for t in byM[-q:]]))
            byS = sorted(tt, key=lambda t: -S[f'{t}_{f}']); ts.append(np.mean([S[f'{t}_{f}'] for t in byS[:q]]) - np.mean([S[f'{t}_{f}'] for t in byS[-q:]]))
        fq = []
        for t in ('FACT', 'SCIQ'):
            lf, T = unit_table(C, m, t, F15, F16, cs, countries)
            if 'BARE' not in T: continue
            ok, sv = T['BARE']
            if ok.sum() < 30: continue
            l, s = lf[ok], sv[ok]; q1, q3 = np.percentile(l, [25, 75]); fq.append(s[l >= q3].mean() - s[l <= q1].mean())
        R[m] = {'format': float(np.mean(fmt)), 'task_by_margin': float(np.median(tm)), 'task_by_survival': float(np.median(ts)), 'fact': float(np.mean(fq))}
    summ = {k: {'median': float(np.median([R[m][k] for m in R])), 'min': float(min(R[m][k] for m in R)), 'max': float(max(R[m][k] for m in R)),
                'median_per_step': float(np.median([R[m][k] for m in R]) / 3)} for k in next(iter(R.values()))}
    json.dump({'per_model': R, 'summary': summ}, open('analysis24_explore.json', 'w'), indent=1)
    for k, v in summ.items(): print(f"{k:18s} quarter gap {v['median']:.3f} ({v['min']:.3f}-{v['max']:.3f})  per step {v['median_per_step']:.3f}")
if __name__ == '__main__': main()
