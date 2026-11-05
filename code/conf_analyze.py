"""Analysis per CONFIDENCE_PLAN.md. python3 conf_analyze.py"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = json.load(open('consolidation.json')); out = {}
def spear(a, b):
    r = lambda v: np.argsort(np.argsort(v)).astype(float); return float(np.corrcoef(r(np.array(a)), r(np.array(b)))[0, 1])
def r2fit(X, y):
    X = np.c_[np.ones(len(y)), X]; be, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ be; return be, float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())
fig, ax = plt.subplots(2, 2, figsize=(13, 11))
for i, name in enumerate(('qwen', 'mistral')):
    P = json.load(open(f'conf_{name}.json')); T = C[name]['T']; S = C[name]['S']
    for j, k in enumerate(('noise', 'ld_rand')):
        n = C[name][k]['n']; ms = [m for m in n if m in P]; y = np.log([n[m] for m in ms]); p1 = np.log(np.maximum([P[m]['P1'] for m in ms], 1e-3)); p2 = np.array([P[m]['P2'] for m in ms])
        b, r2 = r2fit(p1, y); _, r2p2 = r2fit(p2, y); _, r2ts = r2fit(np.c_[np.log([T[m] for m in ms]), [S[m] for m in ms]], y); _, r2all = r2fit(np.c_[p1, np.log([T[m] for m in ms]), [S[m] for m in ms]], y)
        gen = [q for q, m in enumerate(ms) if not m.endswith('CHOICE')]; bg, r2g = r2fit(p1[gen], y[gen])
        out[f'{name}/{k}'] = {'n_measures': len(ms), 'R2_lnP1': round(r2, 3), 'slope': round(float(b[1]), 3), 'spearman_n_P1': round(spear(y, p1), 3), 'R2_P2': round(r2p2, 3), 'spearman_n_P2': round(spear(y, p2), 3),
                              'R2_T_S_only': round(r2ts, 3), 'R2_P1_plus_T_S': round(r2all, 3), 'generative_only': {'n': len(gen), 'R2_lnP1': round(r2g, 3), 'slope': round(float(bg[1]), 3)}}
        a = ax[i, j]; a.scatter(np.exp(p1), np.exp(y), c=['#2471a3' if m.endswith('CHOICE') else '#c0392b' for m in ms]); [a.annotate(m.replace('_', ' '), (np.exp(x), np.exp(v)), fontsize=6) for m, x, v in zip(ms, p1, y)]
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('intact-model uncertainty of the correct output (nats)'); a.set_ylabel('fragility exponent n'); a.set_title(f'{name} / {k}: R2 {r2:.2f} (generative only {r2g:.2f})', fontsize=10); a.grid(alpha=.3)
    out[f'{name}/P'] = {m: [round(P[m]['P1'], 4), round(P[m]['P2'], 2), P[m]['n_ok']] for m in P}
cells = [k for k in out if not k.endswith('/P')]
ok = all(out[k]['R2_lnP1'] >= 0.6 and out[k]['slope'] > 0 for k in cells); fail = sum(out[k]['R2_lnP1'] < 0.3 for k in cells) >= 3
out['verdict'] = 'HOLDS' if ok else ('FAILS' if fail else 'PARTIAL')
fig.suptitle('Does intact-model confidence predict fragility? (red = generative, blue = forced choice)', fontsize=11); fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig('confidence.png', dpi=110)
json.dump(out, open('confidence.json', 'w'), indent=1)
print('VERDICT', out['verdict'])
for k in cells: print(k, out[k])
for name in ('qwen', 'mistral'): print(name, 'P1 (nats), P2 (logit gap), n_ok:', dict(sorted(out[f'{name}/P'].items(), key=lambda z: -z[1][0])))
