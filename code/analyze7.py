"""Locked analysis for PROTOCOL_ADDENDUM_6 (format compliance). Written before any main result was opened. python3 analyze7.py"""
import json, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
K = ['noise', 'quant', 'ld_rand']; TASKS = ['FACT', 'ARITH1', 'COPY']; FF = ['F1', 'F2', 'F3', 'F4', 'F5']
def ranks(v):
    v = np.array([1e9 if (x is None or np.isinf(x)) else x for x in v]); o = v.argsort(); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def sp(a, b):
    ra, rb = ranks(a), ranks(b); return round(float(np.corrcoef(ra, rb)[0, 1]), 3) if ra.std() > 0 and rb.std() > 0 else None
def analyse(d):
    R = {}
    for f in glob.glob(f'{d}/fmt_*.json'):
        x = json.load(open(f)); c = x['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = x
    b = R[('none', 0, 0)]
    dose = {'noise': sorted({k[1] for k in R if k[0] == 'noise'}), 'quant': sorted({k[1] for k in R if k[0] == 'quant'}, reverse=True), 'ld_rand': sorted({k[1] for k in R if k[0] == 'ld_rand'})}
    seeds = {k: sorted({x[2] for x in R if x[0] == k}) for k in K}
    fb = {m: np.mean(v['format']) for m, v in b['fmt'].items()}; cb = {m: np.mean(v['content']) for m, v in b['fmt'].items()}
    use_f = [m for m in fb if fb[m] >= 0.8 and cb[m] >= 0.8]
    rb = {m: (np.mean(v['correct']), v['chance']) for m, v in b['ref'].items()}; use_r = [m for m in rb if rb[m][0] - rb[m][1] >= 0.3]
    def cur(kind, m, k):
        def one(x, s):
            e = R[(k, x, s)]
            if kind == 'format': return np.mean(e['fmt'][m]['format']) / fb[m]
            if kind == 'content': return np.mean(e['fmt'][m]['content']) / cb[m]
            return (np.mean(e['ref'][m]['correct']) - rb[m][1]) / (rb[m][0] - rb[m][1])
        return np.mean([[one(x, s) for x in dose[k]] for s in seeds[k]], 0)
    def d50(k, cs):
        xs = np.r_[0, dose[k]] if k == 'ld_rand' else np.arange(len(cs) + 1); cs = np.r_[1.0, cs]
        for i in range(1, len(cs)):
            if cs[i] < 0.5: return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (cs[i - 1] - 0.5) / (cs[i - 1] - cs[i])) if cs[i - 1] > cs[i] else float(xs[i])
        return np.inf
    fm = [f'{t}_{f}' for t in TASKS for f in FF if f'{t}_{f}' in use_f]
    DF = {k: {m: d50(k, cur('format', m, k)) for m in fm} for k in K}
    DC0 = {k: {t: d50(k, cur('content', f'{t}_F0', k)) for t in TASKS if f'{t}_F0' in use_f} for k in K}
    DL = {k: d50(k, cur('content', 'COPYLONG', k)) if 'COPYLONG' in use_f else None for k in K}
    DR = {k: {m: d50(k, cur('ref', m, k)) for m in use_r} for k in K}
    g1 = {f'{k}/{m}': bool(DF[k][m] < DC0[k][m.split('_')[0]]) for k in K for m in fm if m.split('_')[0] in DC0[k]}
    g2 = {}
    for k in K:
        mn = min(DR[k], key=lambda z: DR[k][z]); n = sum(DF[k][m] < DR[k][mn] for m in fm); g2[k] = {'most_fragile_task': mn, 'its_D50': round(DR[k][mn], 2), 'format_below': int(n), 'of': len(fm), 'holds': bool(n >= len(fm) / 2)}
    g3 = {}
    for k in K:
        for m in fm:
            cf = cur('format', m, k); cc = cur('content', m, k); idx = [i for i, v in enumerate(cf) if v < 0.5]
            if idx: g3[f'{k}/{m}'] = {'format': round(float(cf[idx[0]]), 2), 'content': round(float(cc[idx[0]]), 2), 'holds': bool(cc[idx[0]] >= 0.8)}
    g4 = {k: {'median_format_D50': round(float(np.median([min(DF[k][m], 99) for m in fm])), 2), 'COPYLONG_D50': None if DL[k] is None or np.isinf(DL[k]) else round(DL[k], 2), 'holds': bool(DL[k] is not None and np.median([min(DF[k][m], 99) for m in fm]) < min(DL[k], 99))} for k in K}
    allm = [('format', m) for m in fm] + [('content', f'{t}_F0') for t in TASKS if f'{t}_F0' in use_f] + ([('content', 'COPYLONG')] if 'COPYLONG' in use_f else []) + [('ref', m) for m in use_r]
    DA = {k: [d50(k, cur(kind, m, k)) for kind, m in allm] for k in K}
    g5 = {f'{a}-{b_}': sp(DA[a], DA[b_]) for a, b_ in itertools.combinations(K, 2)}
    fam = {k: {'familiar_F1F2': round(float(np.median([min(DF[k][m], 99) for m in fm if m[-2:] in ('F1', 'F2')])), 2), 'arbitrary_F3F4F5': round(float(np.median([min(DF[k][m], 99) for m in fm if m[-2:] in ('F3', 'F4', 'F5')])), 2)} for k in K}
    rd = lambda v: None if (v is None or np.isinf(v)) else round(v, 2)
    return {'doses': dose, 'usable_format_cells': fm, 'excluded_cells': {m: [round(float(fb[m]), 2), round(float(cb[m]), 2)] for m in fb if m not in use_f}, 'baseline_format': {m: round(float(v), 3) for m, v in fb.items()}, 'baseline_content': {m: round(float(v), 3) for m, v in cb.items()},
            'D50_format': {k: {m: rd(v) for m, v in DF[k].items()} for k in K}, 'D50_content_F0': {k: {t: rd(v) for t, v in DC0[k].items()} for k in K}, 'D50_COPYLONG': {k: rd(DL[k]) for k in K}, 'D50_ref': {k: {m: rd(v) for m, v in DR[k].items()} for k in K},
            'curves_format': {k: {m: [round(float(x), 2) for x in cur('format', m, k)] for m in fm} for k in K}, 'curves_content_same_format': {k: {m: [round(float(x), 2) for x in cur('content', m, k)] for m in fm} for k in K},
            'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4, 'G5': g5, 'familiar_vs_arbitrary': fam, 'labels': [f'{kind}:{m}' for kind, m in allm], 'D50_all': {k: [rd(x) for x in DA[k]] for k in K}}
res = {n: analyse(d) for n, d in [('qwen', 'fmt_qwen'), ('mistral', 'fmt_mis')]}
g1 = [v for r in res.values() for v in r['G1'].values()]; g3 = [v['holds'] for r in res.values() for v in r['G3'].values()]
summ = {'G1_fraction': round(sum(g1) / len(g1), 3), 'G1_n': len(g1), 'G1': sum(g1) / len(g1) >= 0.8,
        'G2_cells': {f'{n}/{k}': r['G2'][k]['holds'] for n, r in res.items() for k in K}, 'G3_fraction': round(sum(g3) / max(1, len(g3)), 3), 'G3_n': len(g3), 'G3': bool(g3 and sum(g3) / len(g3) >= 0.75),
        'G4_cells': {f'{n}/{k}': r['G4'][k]['holds'] for n, r in res.items() for k in K}, 'G5': {n: r['G5'] for n, r in res.items()}}
summ['G2'] = sum(summ['G2_cells'].values()) >= 5; summ['G4'] = sum(summ['G4_cells'].values()) >= 5; summ['G5_pass'] = all(min((v or 0) for v in r['G5'].values()) >= 0.7 for r in res.values())
json.dump({'summary': summ, **res}, open('fmt_analysis7.json', 'w'), indent=1, ensure_ascii=False)
print('SUMMARY', json.dumps(summ, ensure_ascii=False))
for n, r in res.items():
    print('=====', n, '| excluded cells (format, content baselines):', r['excluded_cells'])
    for k in K:
        print(' --', k, r['doses'][k], '| G2', r['G2'][k], '| G4', r['G4'][k], '| fam', r['familiar_vs_arbitrary'][k])
        print('    content F0 D50', r['D50_content_F0'][k], '| COPYLONG', r['D50_COPYLONG'][k])
        print('    ref D50', r['D50_ref'][k])
        for m in sorted(r['usable_format_cells'], key=lambda z: 1e9 if r['D50_format'][k][z] is None else r['D50_format'][k][z]): print(f"      {m:10s} fmt D50={r['D50_format'][k][m]}  format {r['curves_format'][k][m]}  content {r['curves_content_same_format'][k][m]}")
fig, ax = plt.subplots(2, 3, figsize=(20, 11))
for i, (n, r) in enumerate(res.items()):
    for j, k in enumerate(K):
        a = ax[i, j]; lab = r['labels']; v = [99 if x is None else x for x in r['D50_all'][k]]; o = np.argsort(v); cap = max([x for x in v if x < 99] + [1]) * 1.1
        col = ['#c0392b' if lab[t].startswith('format') else '#7f8c8d' if 'COPYLONG' in lab[t] else '#2471a3' if lab[t].startswith('content') else '#27ae60' for t in o]
        a.barh(range(len(o)), [min(v[t], cap) for t in o], color=col); a.set_yticks(range(len(o))); a.set_yticklabels([lab[t].split(':')[1] + (' (fmt)' if lab[t].startswith('format') else '') for t in o], fontsize=6); a.invert_yaxis()
        a.set_title(f'{n} / {k}: D50 (short bar = fails first)', fontsize=9); a.grid(alpha=.3, axis='x')
fig.suptitle('Addendum 6: red = format compliance, blue = content under default format, grey = long-copy control, green = task measures', fontsize=11); fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig('fmt_compliance.png', dpi=110)
