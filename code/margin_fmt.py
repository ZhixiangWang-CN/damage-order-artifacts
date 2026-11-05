"""PROTOCOL_ADDENDUM_13: format-conditioned answer margin on intact models.  python3 margin_fmt.py <items_dir> <model_key> <out.json.gz>
For every eval item, task and generated format f (BARE, JSON, TWICE): summed log p of each candidate answer written in format f
(the candidate in place of the gold in atlas_full.target, then the turn-end token) under the prompt the atlas used for f, chat(body(it, f)).
Candidate 0 is the gold answer. For BARE this is the same prompt and continuation as the addendum-8 CHOICE scores (a consistency check).
"""
import sys, gzip, json
import atlas_full as A
import atlas_comp  # noqa: F401  (patches atlas_full: q32 loading)
from push import score

def fmt_target(c, f): return c if f == 'BARE' else ('{"answer": "%s"}' % c if f == 'JSON' else f'{c}, {c}')

def main(items_dir, key, out_f):
    items, _ = A.load_items(items_dir); tok, model, end = A.load_model(key); res = {}
    for t in A.TASKS:
        its = items[t]['eval']
        for f in ['BARE', 'JSON', 'TWICE']:
            pairs, idx = [], []
            for i, it in enumerate(its):
                p = tok.encode(A.chat(tok, A.body(it, f)), add_special_tokens=False)
                for c in it['cands']: pairs.append((p, tok.encode(fmt_target(c, f), add_special_tokens=False) + [end])); idx.append(i)
            sc = score(model, tok, pairs); per = [[] for _ in its]
            for i, v in zip(idx, sc): per[i].append(round(v, 4))
            res[f'{t}_{f}'] = per
        print(key, t, {f: round(-sum(s[0] - max(s[1:]) for s in res[f'{t}_{f}']) / len(its), 2) for f in ['BARE', 'JSON', 'TWICE']}, flush=True)
    with gzip.open(out_f, 'wt') as fh: json.dump({'model': key, 'res': res}, fh)

if __name__ == '__main__': main(*sys.argv[1:])
