"""Locked analysis for PROTOCOL_ADDENDUM_13 (format-conditioned answer margin). Written before any format-conditioned margin was measured.
  python3 analyze14.py <fmargin_dir> <push_dir> <analysis10_8models.json> <intact_root> [...]   -> analysis14.json + printed verdict
fmargin_m = -mean over all eval items of [score(gold in format f) - best distractor in format f] under the format-f prompt (margin_fmt.py).
Measures, observed exponents and the addendum-12 predictor (z(push) + z(format-free margin)) exactly as analyze13.rows_for.
"""
import sys, os, gzip, json
import numpy as np
import analyze13 as P

MODELS = ['q15', 'llama', 'phi', 'q7', 'mis', 'gemma', 'q14', 'q32']; F3 = P.F3
def pear(a, b): return P.pear(a, b)

def main(fdir, push_dir, a10, roots):
    ex = json.load(open(a10))['exponents']; out = {'per_model': {}}
    for m in MODELS:
        f = f'{fdir}/{m}.json.gz'
        if not os.path.exists(f): continue
        rows = P.rows_for(m, push_dir, ex, roots)
        if not rows: continue
        F = json.load(gzip.open(f, 'rt'))['res']
        for r in rows: r['fmargin'] = -float(np.mean([s[0] - max(s[1:]) for s in F[r['measure']]]))
        y = np.array([r['obs'] for r in rows]); fm = np.array([r['fmargin'] for r in rows])
        pred12 = P.z(np.array([r['push'] for r in rows])) + P.z(np.array([r['margin'] for r in rows]))
        means = {g: (float(np.mean([r['fmargin'] for r in rows if r['format'] == g])), float(np.mean([r['obs'] for r in rows if r['format'] == g]))) for g in F3 if any(r['format'] == g for r in rows)}
        fs = list(means); order_ok = len(fs) == 3 and sorted(fs, key=lambda g: means[g][0]) == sorted(fs, key=lambda g: means[g][1])
        # descriptive: does the per-task increase of the format-conditioned margin over BARE track the per-task extra fragility of the format?
        byt = {}
        for r in rows: byt.setdefault(r['measure'].rsplit('_', 1)[0], {})[r['format']] = r
        extra = {g: pear([v[g]['fmargin'] - v['BARE']['fmargin'] for v in byt.values() if g in v and 'BARE' in v],
                         [v[g]['obs'] - v['BARE']['obs'] for v in byt.values() if g in v and 'BARE' in v]) for g in ('JSON', 'TWICE')}
        out['per_model'][m] = {'n': len(rows), 'Q1_r_fmargin': pear(fm, y), 'r_pred12': pear(pred12, y), 'r_margin_bare_only': pear(np.array([r['margin'] for r in rows]), y),
                               'Q2_format_order_ok': bool(order_ok), 'format_means_fmargin_obs': means, 'Q3_beats_pred12': (pear(fm, y) or -9) > (pear(pred12, y) or -9),
                               'r_z_push_plus_z_fmargin': pear(P.z(np.array([r['push'] for r in rows])) + P.z(fm), y), 'extra_vs_bare_r': extra, 'rows': rows}
    V = out['per_model']; q1 = P.med([v['Q1_r_fmargin'] for v in V.values()])
    s = {'models': len(V), 'Q1_median_r': q1, 'Q1_by_model': {m: v['Q1_r_fmargin'] for m, v in V.items()},
         'Q2_order_exact': sum(v['Q2_format_order_ok'] for v in V.values()), 'Q3_beats_pred12': sum(v['Q3_beats_pred12'] for v in V.values()),
         'median_r_pred12': P.med([v['r_pred12'] for v in V.values()]), 'median_r_margin_bare_only': P.med([v['r_margin_bare_only'] for v in V.values()]),
         'median_extra_vs_bare_r': {g: P.med([v['extra_vs_bare_r'][g] for v in V.values()]) for g in ('JSON', 'TWICE')}}
    a, b, c = q1 or -9, s['Q2_order_exact'], s['Q3_beats_pred12']
    if len(V) < 8: s['verdict'] = f'INCOMPLETE: {len(V)}/8 models'
    elif a >= 0.80 and b >= 7 and c >= 6: s['verdict'] = 'COMMIT: format-conditioned answer margin enters the main line as the account of format fragility'
    elif a >= 0.65 and b >= 6 and c >= 5: s['verdict'] = 'PARTIAL: reported as a partial account (format interference with the answer)'
    else: s['verdict'] = 'STOP: reported as a negative result in the appendix'
    out['summary'] = s; json.dump(out, open('analysis14.json', 'w'), indent=1, default=float); print(json.dumps(s, indent=1, default=float))

if __name__ == '__main__': main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
