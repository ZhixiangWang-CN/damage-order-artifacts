"""Server smoke test for margins.py (a handful of items; not part of the collection)."""
import sys, numpy as np
import margins as G, atlas_full as A
items, _ = A.load_items('items'); tok, model, end = A.load_model(sys.argv[1])
for t in ('ARITH2', 'FACT', 'MMLU_MEM', 'BOOLQ'):
    its = items[t]['eval'][:4]
    for f in G.GEN:
        ps = [A.chat(tok, A.body(it, f)) for it in its]; ref = A.generate(model, tok, ps, A.GEN_MX[f]); g = G.generate_gaps(model, tok, ps, A.GEN_MX[f])
        for it, r, (ids, gp) in zip(its, ref, g):
            s = tok.decode(ids, skip_special_tokens=True); c = A.judge(it, f, s)[0]
            print(t, f, 'same' if s == r else 'DIFF', repr(s)[:40], 'correct', c, 'steps', len(ids), 'crit', G.crit_steps(tok, it, f, ids, c), 'min_gap', round(min(x[0] for x in gp), 2), 'last_tok_is_stop', ids[-1] in ([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int) else model.generation_config.eos_token_id))
