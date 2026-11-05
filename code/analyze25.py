"""Addendum 21: credible baselines and stricter holdouts for prediction levels 3-5 (existing data only).
usage: python3 analyze25.py smoke   # reproduction checks and set sizes only, no new comparison is computed
       python3 analyze25.py run     # locked analysis -> analysis25.json

Accuracy level (levels 4 and 5). Every exponent source below is plugged into the same predictor (analyze19): one strength h per
compression condition, estimated by least squares from calibration measures of that same condition, then
accuracy = chance + (intact - chance) * exp(-n h). All sources share the measure set S_m and the calibration measures, so the
calibration budget is identical. Sources:
  P   random-damage profile of the target model (analyze19.random_exponents)                         [level 4]
  I   intact-only: margin slope and format effects of the other models (analyze21)                    [level 5]
  B1  format only: exp(format effect averaged over the other models)
  B2  task identity: exp(mean task effect of the same task in the other models + format effect)
  B3  intact accuracy: log n = g * z(normalized intact accuracy of the measure); g pooled over the other models
  U   uniform decline (every exponent 1)
Calibration schemes:
  LOMO  leave one measure out, every other measure of S_m in the window calibrates (as analyze19/21; >= 5 measures)
  FIX   a fixed set of four calibration tasks (all their formats) calibrates; every measure of every other task is predicted,
        so whole tasks are held out with all their formats (>= 3 calibration measures in the window). The four tasks are
        at ranks 2, 6, 10 and 14 of the median z-scored intact margin over the eight models (intact data only).
  FAM   FIX, and the sources that borrow from other models (I, B1, B2, B3) exclude every model of the target's family
        (the four Qwen2.5 models form one family; each other model is its own family).
Fact level (level 3): intact item margin (CHOICE score of the correct candidate minus the best distractor, same items as BARE)
as a baseline for which facts compression removes, and the partial Spearman of random-damage survival with compression
survival holding both log frequency and intact margin constant.
"""
import sys, json, pickle
import numpy as np
from scipy.stats import spearmanr
from analyze9 import additive, LO, HI, TASKS
from analyze15 import MODELS, RANDOM, COMPRESS, CACHE, fact_countries
from analyze17 import surv
from analyze19 import cval, random_exponents
from analyze21 import margins

FAMILY = {'q15': 'qwen', 'q7': 'qwen', 'q14': 'qwen', 'q32': 'qwen', 'llama': 'llama', 'mis': 'mis', 'gemma': 'gemma', 'phi': 'phi'}
SOURCES = ['P', 'I', 'B1', 'B2', 'B3', 'U']
split = lambda x: x.rsplit('_', 1)

def base_of(C, m): return C[m]['cond'][('none', 0.0, 0)]

def norm_acc(base, x):
    a0, k = base['res'][x]['acc'], base['res'][x]['chance']; return (a0 - k) / (1 - k)

def borrowed(C, m, MG, AD, EX, others):
    """exponent sources that borrow regularities from the models in `others` (never from m's damaged data)."""
    X, Y, B, TA = [], [], {}, {}
    for o in others:
        if AD[o] is None: continue
        ts = [t for t in AD[o]['a'] if t in MG[o]]; mc, sd = np.mean(list(MG[o].values())), np.std(list(MG[o].values()))
        X += [(MG[o][t] - mc) / sd for t in ts]; Y += [AD[o]['a'][t] for t in ts]
        for f, v in AD[o]['b'].items(): B.setdefault(f, []).append(v)
        for t, v in AD[o]['a'].items(): TA.setdefault(t, []).append(v)
    X, Y = np.array(X), np.array(Y); slope = float((X * Y).sum() / (X * X).sum()); beta = {f: float(np.mean(v)) for f, v in B.items()}
    ta = {t: float(np.mean(v)) for t, v in TA.items()}
    # B3: centred log random-damage exponents of the other models on their z-scored normalized intact accuracy
    Xa, Ya = [], []
    for o in others:
        bo = base_of(C, o); ks = [x for x in EX[o] if norm_acc(bo, x) > 0]
        if len(ks) < 5: continue
        z = np.array([norm_acc(bo, x) for x in ks]); z = (z - z.mean()) / z.std(); y = np.log([EX[o][x] for x in ks]); y = y - y.mean()
        Xa += list(z); Ya += list(y)
    Xa, Ya = np.array(Xa), np.array(Ya); g = float((Xa * Ya).sum() / (Xa * Xa).sum())
    return slope, beta, ta, g

def sources(C, m, MG, AD, EX, others):
    base = base_of(C, m); slope, beta, ta, g = borrowed(C, m, MG, AD, EX, others)
    P = EX[m]
    cand = [x for x in sorted(C[m]['usable']) if base['res'][x]['acc'] - base['res'][x]['chance'] > 0 and split(x)[0] in MG[m] and split(x)[1] in beta]
    S = [x for x in cand if x in P]                                           # the shared measure set S_m
    mc, sd = np.mean(list(MG[m].values())), np.std(list(MG[m].values()))
    za = np.array([norm_acc(base, x) for x in S]); za = (za - za.mean()) / za.std(); za = dict(zip(S, za))
    src = {'P': {x: P[x] for x in S},
           'I': {x: float(np.exp(slope * (MG[m][split(x)[0]] - mc) / sd + beta[split(x)[1]])) for x in S},
           'B1': {x: float(np.exp(beta[split(x)[1]])) for x in S},
           'B2': {x: float(np.exp(ta.get(split(x)[0], 0.0) + beta[split(x)[1]])) for x in S},
           'B3': {x: float(np.exp(g * za[x])) for x in S},
           'U': {x: 1.0 for x in S}}
    return S, src, {'slope': slope, 'g': g}

def predict(base, e, S, src, scheme, cal_tasks):
    """rows (measure, observed, {source: predicted}) for one compression condition"""
    c = {x: cval(base, e, x) for x in S}; rows = []
    if scheme == 'LOMO':
        targets = S; calib = lambda t: [x for x in S if x != t and LO < c[x] < HI]; need = 5
    else:
        targets = [x for x in S if split(x)[0] not in cal_tasks]; cal = [x for x in S if split(x)[0] in cal_tasks and LO < c[x] < HI]
        calib = lambda t: cal; need = 3
    for t in targets:
        use = calib(t)
        if len(use) < need: continue
        L = np.array([-np.log(c[x]) for x in use]); a0, k = base['res'][t]['acc'], base['res'][t]['chance']; pr = {}
        for s, ex in src.items():
            n = np.array([ex[x] for x in use]); h = float((n * L).sum() / (n * n).sum())
            pr[s] = float(k + (a0 - k) * np.exp(-ex[t] * h))
        rows.append((t, float(e['res'][t]['acc']), pr))
    return rows

def calibration_tasks(C):
    """intact data only: rank the 16 tasks by their median (over the eight models) within-model z-scored intact margin and take the
    tasks at ranks 2, 6, 10 and 14, so the calibration set spans fragile to robust tasks."""
    MG = {m: margins(C, m) for m in MODELS}; z = {}
    for m in MODELS:
        v = np.array(list(MG[m].values())); mc, sd = v.mean(), v.std()
        for t, x in MG[m].items(): z.setdefault(t, []).append((x - mc) / sd)
    med = {t: float(np.median(v)) for t, v in z.items()}; order = sorted(med, key=lambda t: (med[t], TASKS.index(t)))
    return [order[i] for i in (1, 5, 9, 13)], med

def accuracy_level(C, scheme, cal_tasks, MG, AD, EX):
    cells = {}
    for m in MODELS:
        others = [o for o in MODELS if o != m and (scheme != 'FAM' or FAMILY[o] != FAMILY[m])]
        S, src, info = sources(C, m, MG, AD, EX, others)
        cond, base = C[m]['cond'], base_of(C, m)
        for k in COMPRESS:
            rows, nc = [], 0
            for key in sorted(cond):
                if key[0] != k or not (0.10 <= np.median([cval(base, cond[key], x) for x in S]) <= 0.90): continue
                nc += 1; rows += predict(base, cond[key], S, src, scheme, cal_tasks)
            if nc == 0 or len(rows) < 20: cells[f'{m}|{k}'] = None; continue
            o = np.array([r[1] for r in rows])
            cells[f'{m}|{k}'] = {'n_conditions': nc, 'n_points': len(rows), 'n_S': len(S), **info,
                                 'mae': {s: float(np.abs(np.array([r[2][s] for r in rows]) - o).mean()) for s in SOURCES}}
    return cells

def compare(cells, a, b):
    """per method: median reduction of MAE of a relative to b, share of cells where a is better"""
    out = {}
    for k in COMPRESS:
        cs = [c for key, c in cells.items() if key.endswith('|' + k) and c]
        red = [1 - c['mae'][a] / c['mae'][b] for c in cs]
        out[k] = {'n_cells': len(cs), 'median_reduction': float(np.median(red)) if cs else None,
                  'frac_better': float(np.mean([c['mae'][a] < c['mae'][b] for c in cs])) if cs else None}
    return out

def med_mae(cells):
    return {k: {s: float(np.median([c['mae'][s] for key, c in cells.items() if key.endswith('|' + k) and c])) for s in SOURCES} for k in COMPRESS}

def verdict(n): return 'COMMIT' if n == 3 else ('PARTIAL' if n == 2 else 'STOP')

# ---------------------------------------------------------------- fact level
def partial_spearman2(x, y, zs):
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    Z = np.column_stack([np.ones(len(x))] + [r(z) for z in zs]); res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    ex, ey = res(r(x)), res(r(y))
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else None

def fact_units(C, m, task, F15, F16, cs, countries, drop=()):
    b = base_of(C, m)['res']; ok0 = b[f'{task}_BARE']['correct']; sc = b[f'{task}_CHOICE']['scores']
    mg = np.array([s[0] - max(s[1:]) for s in sc]); sR = surv(C, m, f'{task}_BARE', RANDOM)
    sC = {d: surv(C, m, f'{task}_BARE', [d]) for d in COMPRESS}
    if task == 'FACT':
        keep = [c for c in countries if ok0[cs == c].all() and c not in drop]
        agg = lambda s: np.array([s[cs == c].mean() for c in keep]) if s is not None else None
        lf = np.array([np.log10(F15[c]['cooc'] + 1) for c in keep])
    else:
        ev = json.load(open(f'items_v8/{task}.json'))['eval']; idx = np.where(ok0 == 1)[0]
        agg = lambda s: s[idx] if s is not None else None
        lf = np.array([np.log10(F16[f"{task}|{ev[i]['ans']}"]['count'] + 1) for i in idx])
    return lf, agg(mg), agg(sR), {d: agg(s) for d, s in sC.items()}

def fact_level(C, drop=()):
    F15 = json.load(open('freq15.json'))['counts']; F16 = json.load(open('freq16.json'))['counts']
    cs = np.array(fact_countries()[0]); countries = sorted(set(cs)); cells = {}
    for task in ('FACT', 'SCIQ'):
        for m in MODELS:
            lf, mg, sR, sC = fact_units(C, m, task, F15, F16, cs, countries, drop)
            for d in COMPRESS:
                y = sC[d]
                if y is None or len(sR) < 30 or np.ptp(sR) == 0 or np.ptp(y) == 0: cells[f'{m}|{task}|{d}'] = None; continue
                cells[f'{m}|{task}|{d}'] = {'n': int(len(sR)), 'rho_random': float(spearmanr(sR, y)[0]), 'rho_margin': float(spearmanr(mg, y)[0]),
                                            'rho_freq': float(spearmanr(lf, y)[0]), 'partial_random_given_freq_margin': partial_spearman2(sR, y, [lf, mg])}
    return cells

def fact_summary(cells):
    out = {}
    for d in COMPRESS:
        cs = [c for key, c in cells.items() if key.endswith('|' + d) and c]
        p = [c['partial_random_given_freq_margin'] for c in cs if c['partial_random_given_freq_margin'] is not None]
        out[d] = {'n_cells': len(cs), 'median_rho_random': float(np.median([c['rho_random'] for c in cs])),
                  'median_rho_margin': float(np.median([c['rho_margin'] for c in cs])), 'median_rho_freq': float(np.median([c['rho_freq'] for c in cs])),
                  'frac_random_beats_margin': float(np.mean([c['rho_random'] > c['rho_margin'] for c in cs])),
                  'median_partial': float(np.median(p)), 'frac_partial_positive': float(np.mean([x > 0 for x in p]))}
        out[d]['pass'] = bool(len(p) >= 8 and out[d]['median_partial'] >= 0.15 and out[d]['frac_partial_positive'] >= 0.75)
    return out

# ---------------------------------------------------------------- main
def prepare(C):
    MG = {m: margins(C, m) for m in MODELS}; EX = {m: random_exponents(C, m) for m in MODELS}; AD = {m: additive(EX[m]) for m in MODELS}
    return MG, AD, EX

def smoke():
    C = pickle.load(open(CACHE, 'rb')); MG, AD, EX = prepare(C)
    cal, cnt = calibration_tasks(C); print('calibration tasks', cal, {t: round(cnt[t], 2) for t in cal})
    for m in MODELS:
        S, src, info = sources(C, m, MG, AD, EX, [o for o in MODELS if o != m])
        print(f'  {m:6s} |S| {len(S)}  calibration measures in S {sum(split(x)[0] in cal for x in S)}  slope {info["slope"]:.3f}  g {info["g"]:.3f}')
    # reproduction: slopes equal analysis21; the shared-set LOMO MAE of P and I lie near analysis19/21 (different measure sets, so not equal)
    a21 = json.load(open('analysis21.json'))
    for m in MODELS:
        s = sources(C, m, MG, AD, EX, [o for o in MODELS if o != m])[2]['slope']; assert abs(s - a21['slopes'][m]) < 1e-9, (m, s)
    print('slopes reproduce analysis21')

def run():
    C = pickle.load(open(CACHE, 'rb')); MG, AD, EX = prepare(C); cal, cnt = calibration_tasks(C)
    R = {'calibration_tasks': cal, 'task_median_margin_z': cnt, 'schemes': {}}
    for sch in ('LOMO', 'FIX', 'FAM'):
        cells = accuracy_level(C, sch, cal, MG, AD, EX)
        R['schemes'][sch] = {'cells': cells, 'median_mae': med_mae(cells),
                             'compare': {f'{a}_vs_{b}': compare(cells, a, b) for a in ('P', 'I') for b in ('B1', 'B2', 'B3', 'U')}}
    # H1: level 4 (P) beats every credible baseline under FIX
    cmpF = R['schemes']['FIX']['compare']; cmpA = R['schemes']['FAM']['compare']
    ok = lambda c, k, thr: bool(c[k]['n_cells'] >= 4 and c[k]['median_reduction'] >= thr and c[k]['frac_better'] >= 0.75)
    R['H1'] = {k: {b: ok(cmpF[f'P_vs_{b}'], k, 0.10) for b in ('B1', 'B2', 'B3')} for k in COMPRESS}
    R['H2'] = {k: {b: ok(cmpA[f'I_vs_{b}'], k, 0.10) for b in ('B1', 'B2', 'B3')} for k in COMPRESS}
    R['H3'] = {k: {'P_vs_U_FIX': ok(cmpF['P_vs_U'], k, 0.25), 'I_vs_U_FAM': ok(cmpA['I_vs_U'], k, 0.20)} for k in COMPRESS}
    for h in ('H1', 'H2', 'H3'): R[h + '_verdict'] = verdict(sum(all(v.values()) for v in R[h].values()))
    fc = fact_level(C); R['fact_cells'] = fc; R['H4'] = fact_summary(fc); R['H4_verdict'] = verdict(sum(R['H4'][d]['pass'] for d in COMPRESS))
    R['H4_sensitivity_drop_Italy_Bulgaria'] = fact_summary(fact_level(C, drop=('Italy', 'Bulgaria')))
    json.dump(R, open('analysis25.json', 'w'), indent=1)
    print('calibration tasks', cal)
    for sch in ('LOMO', 'FIX', 'FAM'):
        print(sch); mm = R['schemes'][sch]['median_mae']
        for k in COMPRESS: print(f'  {k:10s} ' + '  '.join(f'{s} {mm[k][s]:.3f}' for s in SOURCES))
    for h in ('H1', 'H2', 'H3', 'H4'): print(h, R[h + '_verdict'], R[h] if h != 'H4' else {d: R['H4'][d] for d in COMPRESS})

if __name__ == '__main__':
    {'smoke': smoke, 'run': run}[sys.argv[1]]()
