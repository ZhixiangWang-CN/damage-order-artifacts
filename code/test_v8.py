"""Equivalence checks before stage 1 (PROTOCOL_ADDENDUM_8, post-smoke amendments).
  python3 test_v8.py <model_key> <items_dir>
1. prefix-shared forced-choice scores == full-sequence scores (per candidate, and argmax).
2. after dropping layers (re-indexed), cached generation == uncached generation.
"""
import sys, json, time, numpy as np, torch
import atlas_full as A
key, d = sys.argv[1], sys.argv[2]
items, _ = A.load_items(d); tok, model, end = A.load_model(key); dm = A.Dmg(model)
its = [(t, it) for t in A.TASKS for it in items[t]['smoke'][:6]]
P = [tok.encode(A.chat(tok, A.body(it, 'BARE')), add_special_tokens=False) for t, it in its]
C = [[tok.encode(c, add_special_tokens=False) + [end] for c in it['cands']] for t, it in its]
for label, cond in (('intact', None), ('ld_rand 3', {'damage': 'ld_rand', 'dose': 3, 'seed': 1}), ('noise .1', {'damage': 'noise', 'dose': 0.1, 'seed': 1})):
    if cond: dm.apply(dict(cond))
    t0 = time.time(); a = A.choice_scores(model, tok, P, C); t1 = time.time()
    b = A.logprob_sums(model, tok, [(p, c) for p, cs in zip(P, C) for c in cs]); t2 = time.time()
    fa = np.array([x for r in a for x in r]); fb = np.array(b)
    am = np.mean([int(np.argmax(r)) for r in a] == np.array([int(np.argmax(b[k:k + len(cs)])) for k, cs in zip(np.cumsum([0] + [len(c) for c in C])[:-1], C)]))
    print(label, 'CHOICE max|diff|', round(float(np.abs(fa - fb).max()), 4), 'median', round(float(np.median(np.abs(fa - fb))), 4), 'argmax agree', round(float(am), 3), 'time shared/full', round(t1 - t0, 1), round(t2 - t1, 1), flush=True)
    if cond and cond['damage'].startswith('ld_'):
        ps = [A.chat(tok, A.body(it, f)) for t, it in its[:48] for f in ('BARE', 'TWICE')]
        g1 = A.generate(model, tok, ps, 32); model.generation_config.use_cache = False; g2 = A.generate(model, tok, ps, 32); model.generation_config.use_cache = True
        same = sum(x == y for x, y in zip(g1, g2)); print(label, 'cached vs uncached generation identical', same, '/', len(ps), [(x, y) for x, y in zip(g1, g2) if x != y][:3], flush=True)
dm.restore(); print('layers', len(model.model.layers), 'cfg', model.config.num_hidden_layers)
