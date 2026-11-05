"""Locked analysis for PROTOCOL_ADDENDUM_5 (output check). Written before any output was viewed. Runs on the server (needs the datasets)."""
import json, glob, re, random, collections
import numpy as np
import atlas_bench as B
from atlas_out import TASKS
norm = B.norm
sets = {t: B.get_items(t, 200, 'eval') for t in TASKS}
def opt_orders(task):
    r = random.Random(f'opt-{task}-eval'); o = []
    for it in sets[task]:
        opts = list(it['cands']); r.shuffle(opts); o.append(opts)
    return o
ORD = {t: opt_orders(t) for t in TASKS}
LETRE = re.compile(r'^\s*\(?([A-Ea-e])[\.\):]?(?=\s|$)\s*')
def cls(task, fmt, i, out, gold):
    t = out.strip(); it = sets[task][i]
    if fmt == 'GENOPT':
        opts = ORD[task][i]; m = LETRE.match(t); rest = t[m.end():] if m else t; g = norm(gold); o = norm(rest)
        if g and o.startswith(g[:25]): return 'correct_text'
        for w in opts:
            if w != gold and norm(w) and o.startswith(norm(w)[:25]): return 'wrong_text'
        if m:
            k = 'ABCDE'.index(m.group(1).upper())
            return ('correct_letter' if k < len(opts) and opts[k] == gold else 'wrong_letter')
        return 'other'
    if fmt == 'COT':
        m = re.findall(r'answer is[^\d\-]{0,12}(-?\d[\d,]*(?:\.\d+)?)', t.lower())
        if not m: return 'no_final_answer'
        try: return 'correct' if abs(float(m[-1].replace(',', '').rstrip('.')) - float(gold)) < 1e-6 else 'wrong_number'
        except ValueError: return 'no_final_answer'
    if it['kind'] == 'number':
        m = re.search(r'-?\d[\d,]*', t)
        return 'no_number' if not m else ('correct' if m.group(0).replace(',', '') == gold else 'wrong_number')
    if it['kind'] == 'yesno':
        m = re.search(r'\b(yes|no)\b', t.lower()); return 'no_yesno' if not m else ('correct' if m.group(1) == gold else 'wrong_yesno')
    g = norm(gold); o = norm(t)
    if g and o.startswith(g[:25]): return 'correct'
    if g and g in o: return 'contains_gold'
    return 'other_answer' if 0 < len(t.split()) <= 6 else 'non_answer'
FORMAT_FAIL = {'correct_letter', 'wrong_letter', 'other', 'no_final_answer', 'no_number', 'no_yesno', 'non_answer'}
LENIENT_OK = {'correct_text', 'correct_letter', 'correct', 'contains_gold'}
STRICT_OK = {'correct_text', 'correct'}
R = {}
for f in glob.glob('outchk/out_*.json'):
    d = json.load(open(f)); c = d['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = d['eval']
BN = {}
for f in glob.glob('bench/bench_*.json'):
    d = json.load(open(f)); c = d['cond']; BN[(c['damage'], c['dose'], c.get('seed', 0))] = d['eval']
out = {'conditions': [list(k) for k in sorted(R, key=str)], 'counts': {}, 'examples': {}}
diffs = []
for k, ev in R.items():
    for m, v in ev.items():
        task, fmt = m.rsplit('_', 1); cc = collections.Counter(cls(task, fmt, i, o, g) for i, (o, g) in enumerate(zip(v['outputs'], v['gold'])))
        out['counts'].setdefault(str(k), {})[m] = dict(cc)
        strict = sum(cc[x] for x in STRICT_OK) / len(v['outputs'])
        if k in BN and m in BN[k]: diffs.append(abs(strict - BN[k][m]['acc']))
        if k[0] != 'none':
            ex = [o[:80] for i, (o, g) in enumerate(zip(v['outputs'], v['gold'])) if cls(task, fmt, i, o, g) in FORMAT_FAIL][:4]
            out['examples'].setdefault(m, {})[str(k)] = ex
out['E0_median_abs_diff'] = round(float(np.median(diffs)), 4); out['E0_max_abs_diff'] = round(float(max(diffs)), 3); out['E0'] = out['E0_median_abs_diff'] <= 0.03
dam = [k for k in R if k[0] != 'none']
def pool(ms):
    c = collections.Counter()
    for k in dam:
        for m in ms: c.update(out['counts'][str(k)][m])
    return c
go = [m for m in next(iter(R.values())) if m.endswith('GENOPT')]; c = pool(go); err = sum(v for x, v in c.items() if x not in STRICT_OK)
out['E1_genopt_error_breakdown'] = {x: round(v / err, 3) for x, v in c.items() if x not in STRICT_OK}; out['E1_format_failure_share'] = round(sum(v for x, v in c.items() if x in FORMAT_FAIL) / err, 3); out['E1'] = out['E1_format_failure_share'] > 0.5
base = R[('none', 0, 0)]; bb = BN[('none', 0, 0)]
def acc(v_counts, ok, n): return sum(v_counts.get(x, 0) for x in ok) / n
ratios = []; cells = {}
for k in dam:
    for m in go:
        t = m.rsplit('_', 1)[0]; ch = bb[m]['chance']; n = len(R[k][m]['outputs']); cm = f'{t}_CHOICE'
        cb = out['counts'][str(('none', 0, 0))][m]; ck = out['counts'][str(k)][m]
        s0, l0 = acc(cb, STRICT_OK, n), acc(cb, LENIENT_OK, n)
        cs = (acc(ck, STRICT_OK, n) - ch) / (s0 - ch); cl = (acc(ck, LENIENT_OK, n) - ch) / (l0 - ch)
        cc = (BN[k][cm]['acc'] - ch) / (bb[cm]['acc'] - ch)
        cells[f'{k}/{t}'] = {'strict': round(cs, 2), 'lenient': round(cl, 2), 'choice': round(cc, 2)}
        if 0.2 <= cc <= 0.9 and abs(cc - cs) > 1e-6: ratios.append((cl - cs) / (cc - cs))
out['E2_cells'] = cells; out['E2_n_cells'] = len(ratios); out['E2_median_gap_closed'] = round(float(np.median(ratios)), 3) if ratios else None; out['E2'] = bool(ratios and np.median(ratios) >= 0.5)
e3 = {}
for m, good, bad in [('ARITH3_GEN', 'wrong_number', 'no_number'), ('GSM8K_COT', 'wrong_number', 'no_final_answer')]:
    c = pool([m]); err = sum(v for x, v in c.items() if x not in STRICT_OK); e3[m] = {'errors': err, 'wrong_number_share': round(c[good] / err, 3) if err else None, 'no_answer_share': round(c[bad] / err, 3) if err else None}
out['E3'] = e3; out['E3_pass'] = all((v['wrong_number_share'] or 0) > 0.5 for v in e3.values())
ex = {}
for m in ['SCIQ_GEN', 'FACT_GEN', 'BOOLQ_GEN']:
    c = pool([m]); err = sum(v for x, v in c.items() if x not in STRICT_OK); ex[m] = {x: round(v / err, 3) for x, v in c.items() if x not in STRICT_OK}
out['explore_error_breakdown'] = ex
bydmg = {}
for dmg in ('noise', 'quant', 'ld_rand'):
    c = collections.Counter()
    for k in dam:
        if k[0] == dmg:
            for m in go: c.update(out['counts'][str(k)][m])
    err = sum(v for x, v in c.items() if x not in STRICT_OK); bydmg[dmg] = {x: round(v / err, 3) for x, v in c.items() if x not in STRICT_OK}
out['genopt_errors_by_damage'] = bydmg
json.dump(out, open('outchk/analysis6.json', 'w'), indent=1, ensure_ascii=False)
for k in ['E0_median_abs_diff', 'E0_max_abs_diff', 'E0', 'E1_genopt_error_breakdown', 'E1_format_failure_share', 'E1', 'E2_n_cells', 'E2_median_gap_closed', 'E2', 'E3', 'E3_pass', 'explore_error_breakdown', 'genopt_errors_by_damage']: print(k, out[k])
print('E2 cells:'); [print('  ', k, v) for k, v in cells.items()]
print('baseline counts:', out['counts'][str(('none', 0, 0))])
for m in go[:2] + ['SCIQ_GEN', 'GSM8K_COT']:
    print('examples of format failures', m)
    for k, v in list(out['examples'].get(m, {}).items())[:4]: print('   ', k, [repr(x) for x in v[:3]])
