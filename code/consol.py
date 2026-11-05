"""Consolidation analysis per CONSOLIDATION_PLAN.md. python3 consol.py"""
import json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SV = json.load(open('consol_server.json')); K = ['noise', 'ld_rand']
SINT = {'COPY': 0, 'COPYLONG': 0, 'FACT': 1, 'RETR': 1, 'ARITH1': 1, 'ARITH2': 2, 'ARITH3': 3}
def spear(a, b):
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(r(np.array(a)), r(np.array(b)))[0, 1])
def load(d):
    R = {}
    for f in glob.glob(f'{d}/fmt_*.json'):
        x = json.load(open(f)); c = x['cond']; R[(c['damage'], c['dose'], c.get('seed', 0))] = x
    return R
def rate(e, m):
    if m in e['fmt']:
        v = e['fmt'][m]; return float(np.mean(np.array(v['format']) * np.array(v['content']))), 0.0
    v = e['ref'][m]; return float(np.mean(v['correct'])), v['chance']
def rank1(L, M, it=500):
    n = np.ones(L.shape[0]); h = np.ones(L.shape[1])
    for _ in range(it):
        h = np.array([(M[:, j] * n * L[:, j]).sum() / max((M[:, j] * n * n).sum(), 1e-12) for j in range(L.shape[1])])
        n = np.array([(M[i] * h * L[i]).sum() / max((M[i] * h * h).sum(), 1e-12) for i in range(L.shape[0])])
    s = np.exp(np.mean(np.log(n[n > 0]))); return n / s, h * s
out = {}
fig, ax = plt.subplots(2, 4, figsize=(22, 10))
for mi, (name, d) in enumerate([('qwen', 'fmt_qwen'), ('mistral', 'fmt_mis')]):
    R = load(d); b = R[('none', 0, 0)]; T = SV[f'T_{name}']; ms = []
    for m in T:
        r0, ch = rate(b, m)
        if (r0 >= 0.8) if ch == 0 else (r0 - ch >= 0.3): ms.append(m)
    S = {m: SINT[m.split('_')[0]] for m in ms}; out[name] = {'measures': ms, 'excluded': [m for m in T if m not in ms], 'T': {m: T[m] for m in ms}, 'S': S}
    for ki, k in enumerate(K):
        doses = sorted({x[1] for x in R if x[0] == k}); seeds = sorted({x[2] for x in R if x[0] == k})
        C = np.array([[np.mean([(rate(R[(k, x, s)], m)[0] - rate(b, m)[1]) / (rate(b, m)[0] - rate(b, m)[1]) for s in seeds]) for x in doses] for m in ms])
        M = ((C > 0.03) & (C < 0.97)).astype(float); L = -np.log(np.clip(C, 1e-6, 1.0))
        keep = M.sum(1) >= 2; n, h = rank1(L[keep], M[keep]); mk = [m for m, q in zip(ms, keep) if q]
        P = np.exp(-np.outer(n, h)); mm = M[keep] > 0; a, p = C[keep][mm], P[mm]; r2 = 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        Tv = np.array([T[m] for m in mk]); Sv = np.array([S[m] for m in mk], float); y = np.log(n)
        def fit(X):
            X = np.c_[np.ones(len(y)), X]; be, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ be; return be, 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
        bf, r2f = fit(np.c_[np.log(Tv), Sv]); bt, r2t = fit(np.log(Tv)); bs, r2s = fit(Sv)
        out[name][k] = {'doses': doses, 'n_measures_fit': len(mk), 'dropped_fewer_than_2_cells': [m for m, q in zip(ms, keep) if not q], 'cells_used': int(mm.sum()), 'R2_rank1': round(float(r2), 3), 'h': [round(float(x), 3) for x in h],
                        'n': {m: round(float(v), 3) for m, v in zip(mk, n)}, 'R2_T_and_S': round(float(r2f), 3), 'coef_lnT': round(float(bf[1]), 3), 'coef_S': round(float(bf[2]), 3), 'R2_T_only': round(float(r2t), 3), 'R2_S_only': round(float(r2s), 3),
                        'spearman_n_T': round(spear(n, Tv), 3), 'spearman_n_S': round(spear(n, Sv), 3)}
        a1 = ax[mi, 2 * ki]; a1.scatter(p, a, s=8, alpha=.6); a1.plot([0, 1], [0, 1], 'k:'); a1.set_xlabel('rank-1 model'); a1.set_ylabel('observed corrected score'); a1.set_title(f'{name} / {k}: one hazard, one exponent per measure (R2 {r2:.2f})', fontsize=9); a1.grid(alpha=.3)
        a2 = ax[mi, 2 * ki + 1]; pred = np.exp(np.c_[np.ones(len(y)), np.log(Tv), Sv] @ bf)
        a2.scatter(pred, n, c=Sv, cmap='viridis', s=30); [a2.annotate(m.replace('_', ' '), (pp, nn), fontsize=5) for m, pp, nn in zip(mk, pred, n)]
        lim = [min(pred.min(), n.min()) * .8, max(pred.max(), n.max()) * 1.2]; a2.plot(lim, lim, 'k:'); a2.set_xscale('log'); a2.set_yscale('log'); a2.set_xlabel('exponent predicted from output tokens T and internal steps S'); a2.set_ylabel('fitted exponent n'); a2.set_title(f'{name} / {k}: R2 {r2f:.2f} (T only {r2t:.2f}, S only {r2s:.2f})', fontsize=9); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig('consolidation.png', dpi=110)
cells = [(m, k) for m in ('qwen', 'mistral') for k in K]
A = all(out[m][k]['R2_rank1'] >= 0.8 for m, k in cells); Bc = all(out[m][k]['R2_T_and_S'] >= 0.6 and out[m][k]['coef_lnT'] > 0 and out[m][k]['coef_S'] > 0 for m, k in cells); fail = sum(out[m][k]['R2_T_and_S'] < 0.3 for m, k in cells) >= 3
out['verdict'] = 'HOLDS' if (A and Bc) else ('FAILS' if fail else 'PARTIAL'); out['analysis1_all_ge_0.8'] = bool(A); out['analysis2_all_ok'] = bool(Bc)
# secondary: GSM8K by number of solution steps
st = np.array(SV['gsm8k_steps']); G = {tuple(k.split('|')): v for k, v in SV['gsm8k'].items()}; grp = {'<=2': st <= 2, '3': st == 3, '4': st == 4, '>=5': st >= 5}; sec = {}
for fmt, nI in (('GSM8K_CHOICE', 200), ('GSM8K_COT', 100)):
    base = np.array(G[('none', '0', '0')][fmt]); s_ = st[:nI]
    for k in K:
        doses = sorted({float(x[1]) for x in G if x[0] == k}); seeds = sorted({x[2] for x in G if x[0] == k}); rows = {}
        for d in doses:
            key = lambda s: (k, str(d) if k == 'noise' else str(int(d)), s)
            acc = np.mean([G[key(s)][fmt] for s in seeds], 0); tot = acc.mean() / base.mean()
            if 0.3 <= tot <= 0.8:
                r = {}
                for g, msk in grp.items():
                    mk_ = msk[:nI]
                    if base[mk_].sum() >= 15: r[g] = round(float(acc[mk_].mean() / base[mk_].mean()), 3)
                rows[str(d)] = r
        vals = [(i, v) for r in rows.values() for i, (g, v) in enumerate(r.items())]
        rho = round(spear([a for a, _ in vals], [b_ for _, b_ in vals]), 3) if len(vals) >= 4 else None
        sec[f'{k}/{fmt}'] = {'by_dose': rows, 'spearman_steps_retention': rho, 'holds': bool(rho is not None and rho < 0), 'baseline_acc_by_group': {g: round(float(base[m_[:nI]].mean()), 2) for g, m_ in grp.items()}, 'n_by_group': {g: int(m_[:nI].sum()) for g, m_ in grp.items()}}
out['secondary_gsm8k'] = sec; out['secondary_pass'] = sum(v['holds'] for v in sec.values()) >= 3
json.dump(out, open('consolidation.json', 'w'), indent=1, ensure_ascii=False)
print('VERDICT', out['verdict'], '| analysis1', A, '| analysis2', Bc)
for m, k in cells:
    o = out[m][k]; print(f"{m}/{k}: R2 rank1 {o['R2_rank1']} ({o['cells_used']} cells, {o['n_measures_fit']} measures; dropped {o['dropped_fewer_than_2_cells']}) | n~T,S R2 {o['R2_T_and_S']} b_lnT {o['coef_lnT']} g_S {o['coef_S']} | T only {o['R2_T_only']} S only {o['R2_S_only']} | rho(n,T) {o['spearman_n_T']} rho(n,S) {o['spearman_n_S']}")
    print('    n:', dict(sorted(o['n'].items(), key=lambda z: -z[1])))
print('excluded', {m: out[m]['excluded'] for m in ('qwen', 'mistral')})
for k, v in sec.items(): print('GSM8K', k, 'rho', v['spearman_steps_retention'], 'holds', v['holds'], '| base acc', v['baseline_acc_by_group'], 'n', v['n_by_group']); [print('     dose', d, r) for d, r in v['by_dose'].items()]
print('secondary_pass', out['secondary_pass'])
