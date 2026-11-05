"""Locked analysis for PROTOCOL_ADDENDUM_11 (realistic tasks). Written before any damaged condition of the new tasks existed.
  python3 analyze12.py <real_root> <orig_root> [<orig_root> ...]   -> analysis12.json + printed verdicts
real_root/<model>/<damage>_<dose>_<seed>.json.gz from atlas_real.py; orig_roots hold the addendum-8/9 conditions (analyze9.load_model).
The damage strength h(d) of every dose is taken from the original cell (analyze9.fit_cell on the original measures, unchanged);
each new measure gets the exponent n = sum h(-log c) / sum h^2 over its points inside the window, on the same normalized scale.
"""
import sys, os, glob, gzip, json
import numpy as np
import analyze9 as Z

MODELS = ['q7', 'llama', 'mis', 'gemma']; DAMS = ['noise', 'wanda']; NEW = ['GSM_COT', 'CODE', 'IFEVAL', 'FCALL']
def pear(a, b): return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else None

def load_real(d):
    R = {}
    for f in glob.glob(f'{d}/*.json.gz'):
        with gzip.open(f, 'rt') as fh: x = json.load(fh)
        c = x['cond']; R[(c['damage'], c['dose'], c['seed'])] = x
    return R

def project(h, accs, a0):
    """accs: dose -> list of accuracies over seeds. Returns (n, points) or (None, points)."""
    pts = [(h[x], float(np.mean(v)) / a0) for x, v in accs.items() if x in h and v]
    pts = [(hh, c) for hh, c in pts if Z.LO < c < Z.HI]
    if len(pts) < 2: return None, pts
    hh = np.array([p[0] for p in pts]); y = -np.log([p[1] for p in pts]); return float((hh * y).sum() / (hh * hh).sum()), pts

def pctl(v, allv): return float(np.mean([x < v for x in allv]))

def main(real_root, orig_roots):
    out = {'checks': {}, 'usable_new': {}, 'exponents_new': {}, 'N1': {}, 'N2': {}, 'N3': {}}
    for m in MODELS:
        RR = load_real(f'{real_root}/{m}')
        if ('none', 0, 0) not in RR: continue
        bad = [k for k, x in RR.items() if not x.get('check_ok', True)]; out['checks'][m] = {'n_conditions': len(RR), 'failed_probe_check': [list(k) for k in bad]}
        RR = {k: v for k, v in RR.items() if k not in bad}
        a0 = {t: RR[('none', 0, 0)]['res'][t]['acc'] for t in NEW}
        a0['FCALL_ANY'] = float(np.mean(RR[('none', 0, 0)]['res']['FCALL']['content_any']))
        use = [t for t in NEW if a0[t] >= 0.4]   # chance 0: the addendum-8 usable rule reduces to intact accuracy >= 0.4
        out['usable_new'][m] = {'intact_acc': {t: round(v, 3) for t, v in a0.items()}, 'usable': use}
        dirs = [f'{r}/{m}' for r in orig_roots if os.path.isdir(f'{r}/{m}')]; R, u = Z.load_model(dirs); base = R[('none', 0, 0)]; ms = sorted(u['usable'])
        EX, NN = {}, {}
        for k in DAMS:
            r = Z.fit_cell(R, base, ms, k)
            if r is None: continue
            ex, h, _ = r; EX[k] = ex; NN[k] = {}
            for t in use + (['FCALL_ANY'] if 'FCALL' in use else []):
                accs = {}
                for (kk, x, s), v in RR.items():
                    if kk != k: continue
                    acc = float(np.mean(v['res']['FCALL']['content_any'])) if t == 'FCALL_ANY' else v['res'][t]['acc']
                    accs.setdefault(x, []).append(acc)
                n, pts = project(h, accs, a0[t]); NN[k][t] = n
        out['exponents_new'][m] = {k: {t: (None if v is None else round(v, 4)) for t, v in d.items()} for k, d in NN.items()}
        # N1: all measures fitted under both damages
        if all(k in EX for k in DAMS):
            old = [q for q in EX['noise'] if q in EX['wanda']]; new = [t for t in use if NN['noise'].get(t) and NN['wanda'].get(t)]
            xs = [np.log(EX['noise'][q]) for q in old] + [np.log(NN['noise'][t]) for t in new]; ys = [np.log(EX['wanda'][q]) for q in old] + [np.log(NN['wanda'][t]) for t in new]
            pos = {t: {k: pctl(NN[k][t], list(EX[k].values()) + [NN[k][x] for x in new if x != t]) for k in DAMS} for t in new}
            out['N1'][m] = {'r_all': pear(xs, ys), 'n_old': len(old), 'n_new': len(new), 'positions': pos,
                            'positions_within_25': {t: abs(p['noise'] - p['wanda']) <= 0.25 for t, p in pos.items()}}
        # N2 (descriptive): where the new measures fall within the original measures of each cell
        out['N2'][m] = {k: {t: (None if NN[k].get(t) is None else round(pctl(NN[k][t], list(EX[k].values())), 3)) for t in use} for k in EX}
        # N3: function calling fails in format first
        if 'FCALL' in use:
            out['N3'][m] = {}
            ok0 = np.array(RR[('none', 0, 0)]['res']['FCALL']['correct']) == 1
            for k in DAMS:
                cs = {}
                for (kk, x, s), v in RR.items():
                    if kk == k: cs.setdefault(x, []).append(v)
                cc = {x: float(np.mean([v['res']['FCALL']['acc'] for v in vs])) / a0['FCALL'] for x, vs in cs.items()}
                mid = [x for x, c in cc.items() if 0.3 <= c <= 0.7]
                res = {'c_by_dose': {str(x): round(c, 3) for x, c in sorted(cc.items())}}
                if mid:
                    x = min(mid, key=lambda z: abs(cc[z] - 0.5)); err, ff, fa = 0, 0, 0
                    for v in cs[x]:
                        r = v['res']['FCALL']
                        for i in np.where(ok0)[0]:
                            if r['correct'][i] == 0:
                                err += 1
                                if r['format_ok'][i] == 0: ff += 1; fa += r['content_any'][i]
                    res.update({'dose': x, 'c': round(cc[x], 3), 'n_errors': err, 'format_failure_share': ff / err if err else None,
                                'content_present_among_format_failures': fa / ff if ff else None})
                ns, nl = NN.get(k, {}).get('FCALL'), NN.get(k, {}).get('FCALL_ANY')
                res['exponent_ratio_strict_over_lenient'] = ns / nl if ns and nl else None
                out['N3'][m][k] = res
    r1 = [v['r_all'] for v in out['N1'].values() if v['r_all'] is not None]; w = [b for v in out['N1'].values() for b in v['positions_within_25'].values()]
    n3 = [v for d in out['N3'].values() for v in d.values()]
    f1 = [v['format_failure_share'] for v in n3 if v.get('format_failure_share') is not None]; f2 = [v['content_present_among_format_failures'] for v in n3 if v.get('content_present_among_format_failures') is not None]
    f3 = [v['exponent_ratio_strict_over_lenient'] for v in n3 if v.get('exponent_ratio_strict_over_lenient') is not None]
    s = {'N1_median_r': Z.med(r1), 'N1_models': len(r1), 'N1_positions_within_25': (float(np.mean(w)) if w else None, len(w)),
         'N3_format_failure_share_median': Z.med(f1), 'N3_content_present_median': Z.med(f2), 'N3_exponent_ratio_median': Z.med(f3), 'N3_cells': len(n3)}
    n1 = s['N1_median_r'] is not None and s['N1_median_r'] >= 0.7 and w and np.mean(w) >= 0.75
    n3ok = all(v is not None for v in (s['N3_format_failure_share_median'], s['N3_content_present_median'], s['N3_exponent_ratio_median'])) and \
        s['N3_format_failure_share_median'] >= 0.5 and s['N3_content_present_median'] >= 0.5 and s['N3_exponent_ratio_median'] >= 1.5
    s['N1_pass'] = bool(n1); s['N3_pass'] = bool(n3ok)
    s['verdict'] = ('NMI: realistic tasks share the order, structured output fails before the answer' if n1 and n3ok else
                    'Nature Communications / NeurIPS: realistic tasks share the order; format-first only for short-answer tasks' if n1 else
                    'TMLR: claim narrowed to short-answer tasks; realistic tasks reported as a limitation of scope')
    out['summary'] = s; json.dump(out, open('analysis12.json', 'w'), indent=1, default=float); print(json.dumps(s, indent=1, default=float))

if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
