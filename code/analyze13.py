"""Locked analysis for PROTOCOL_ADDENDUM_12 (instruction push + answer margin). Written before any push of a held-out model existed.
  python3 analyze13.py <push_dir> <analysis10_8models.json> <intact_root> [<intact_root> ...]   -> analysis13.json + printed verdict
push_dir/<model>.json.gz from push.py; intact roots hold <model>/none_0_0.json.gz (addendum 8; q32 from addendum 9).
Prediction for measure m = task t x format f (f in BARE, JSON, TWICE), with no parameter fitted on damaged data:
  push_m   = mean over intact-correct items of [log p(target | prompt with format instruction) - log p(target | question only)]
  margin_t = -mean over all eval items of [CHOICE score of the correct candidate - best distractor]   (the existing answer-margin predictor)
  pred_m   = z(push) + z(margin), z over the model's analysed measures.
Observed: mean log exponent over the four random damages that fitted the measure (at least two), as analyze11.observed.
"""
import sys, os, gzip, json
import numpy as np

HELD = ['q15', 'phi', 'mis', 'gemma', 'q14', 'q32']; PILOT = ['q7', 'llama']; UNT = ['noise', 'zero', 'neuron', 'ld_rand']; F3 = ['BARE', 'JSON', 'TWICE']
def pear(a, b): return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0 else None
def med(v): v = [x for x in v if x is not None]; return float(np.median(v)) if v else None
def z(a): return (a - a.mean()) / a.std()

def observed(ex, m):
    v = {}
    for k in UNT:
        for mm, x in (ex[m].get(k) or {}).items(): v.setdefault(mm, []).append(np.log(x))
    return {mm: float(np.mean(x)) for mm, x in v.items() if len(x) >= 2}

def rows_for(m, push_dir, ex, roots):
    f = f'{push_dir}/{m}.json.gz'; d = next((f'{r}/{m}' for r in roots if os.path.exists(f'{r}/{m}/none_0_0.json.gz')), None)
    if not os.path.exists(f) or d is None: return None
    P = json.load(gzip.open(f, 'rt'))['res']; base = json.load(gzip.open(f'{d}/none_0_0.json.gz', 'rt'))['res']; obs = observed(ex, m); rows = []
    for ms, p in sorted(P.items()):
        if ms not in obs: continue
        t, fm = ms.rsplit('_', 1); ok = np.array(base[ms]['correct']) == 1
        if ok.sum() < 10: continue
        push = float(np.mean((np.array(p['with']) - np.array(p['without']))[ok]))
        mg = -float(np.mean([s[0] - max(s[1:]) for s in base[f'{t}_CHOICE']['scores']]))
        rows.append({'measure': ms, 'format': fm, 'obs': obs[ms], 'push': push, 'margin': mg})
    return rows

def model_tests(rows):
    y = np.array([r['obs'] for r in rows]); a = np.array([r['push'] for r in rows]); b = np.array([r['margin'] for r in rows]); pred = z(a) + z(b)
    fm = {f: (float(np.mean([r['push'] for r in rows if r['format'] == f])), float(np.mean([r['obs'] for r in rows if r['format'] == f])))
          for f in F3 if any(r['format'] == f for r in rows)}
    fs = list(fm); order_ok = len(fs) == 3 and sorted(fs, key=lambda f: fm[f][0]) == sorted(fs, key=lambda f: fm[f][1])
    # descriptive: push reduced to its format means (a pure between-format label) in place of item-level push
    lab = np.array([fm[r['format']][0] for r in rows])
    within = {f: pear(a[[r['format'] == f for r in rows]], y[[r['format'] == f for r in rows]]) for f in F3 if sum(r['format'] == f for r in rows) >= 6}
    return {'n': len(rows), 'P1_r_pred': pear(pred, y), 'r_margin_only': pear(b, y), 'r_push_only': pear(a, y),
            'P2_format_order_ok': bool(order_ok), 'n_formats': len(fs), 'format_means_push_obs': fm,
            'P3_push_adds': (pear(pred, y) or -9) > (pear(b, y) or -9), 'r_pred_with_format_mean_push': pear(z(lab) + z(b), y), 'within_format_r_push': within}

def main(push_dir, a10, roots):
    ex = json.load(open(a10))['exponents']; out = {'held_out': {}, 'pilot': {}}
    for grp, ms in (('held_out', HELD), ('pilot', PILOT)):
        for m in ms:
            rows = rows_for(m, push_dir, ex, roots)
            if rows: out[grp][m] = model_tests(rows); out[grp][m]['rows'] = rows
    H = out['held_out']; r1 = [v['P1_r_pred'] for v in H.values()]
    s = {'models': len(H), 'P1_median_r': med(r1), 'P1_by_model': {m: v['P1_r_pred'] for m, v in H.items()},
         'P2_order_exact': sum(v['P2_format_order_ok'] for v in H.values()), 'P3_push_adds': sum(v['P3_push_adds'] for v in H.values()),
         'median_r_margin_only': med([v['r_margin_only'] for v in H.values()]), 'median_r_push_only': med([v['r_push_only'] for v in H.values()]),
         'median_r_format_mean_push': med([v['r_pred_with_format_mean_push'] for v in H.values()])}
    p1, p2, p3 = s['P1_median_r'] or -9, s['P2_order_exact'], s['P3_push_adds']
    if len(H) < 6: s['verdict'] = f'INCOMPLETE: {len(H)}/6 held-out models'
    elif p1 >= 0.80 and p2 >= 5 and p3 >= 5: s['verdict'] = 'COMMIT: two-part account (answer margin + instruction push) enters the main line'
    elif p1 >= 0.65 and p2 >= 4 and p3 >= 4: s['verdict'] = 'PARTIAL: reported as a partial two-part account'
    else: s['verdict'] = 'STOP: instruction push reported as a negative result in the appendix'
    out['summary'] = s; json.dump(out, open('analysis13.json', 'w'), indent=1, default=float); print(json.dumps(s, indent=1, default=float))

if __name__ == '__main__': main(sys.argv[1], sys.argv[2], sys.argv[3:])
