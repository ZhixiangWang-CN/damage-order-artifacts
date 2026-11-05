"""PROTOCOL_ADDENDUM_13b: first-answer margin under each format's prompt, intact models.  python3 margin_first.py <items_dir> <model_key> <out.json.gz>
Corrects addendum 13, whose TWICE score summed both copies. For every candidate, score only the first written answer and the token that closes it:
  BARE   candidate + turn end            (identical to addendum 13 and the addendum-8 CHOICE scores)
  JSON   '{"answer": "' + candidate + '"'   (the template prefix is the same for every candidate, so it cancels in the margin)
  TWICE  candidate + ','                 (the first copy and its separator; the second copy is not scored)
under the prompt the atlas used for that format, chat(body(it, f)). Candidate 0 is the gold answer.
"""
import sys, gzip, json
import atlas_full as A
import atlas_comp  # noqa: F401  (patches atlas_full: q32 loading)
from push import score

def first(tok, c, f, end):
    if f == 'BARE': return tok.encode(c, add_special_tokens=False) + [end]
    return tok.encode('{"answer": "%s"' % c if f == 'JSON' else f'{c},', add_special_tokens=False)

def main(items_dir, key, out_f):
    items, _ = A.load_items(items_dir); tok, model, end = A.load_model(key); res = {}
    for t in A.TASKS:
        its = items[t]['eval']
        for f in ['BARE', 'JSON', 'TWICE']:
            pairs, idx = [], []
            for i, it in enumerate(its):
                p = tok.encode(A.chat(tok, A.body(it, f)), add_special_tokens=False)
                for c in it['cands']: pairs.append((p, first(tok, c, f, end))); idx.append(i)
            sc = score(model, tok, pairs); per = [[] for _ in its]
            for i, v in zip(idx, sc): per[i].append(round(v, 4))
            res[f'{t}_{f}'] = per
        print(key, t, {f: round(-sum(s[0] - max(s[1:]) for s in res[f'{t}_{f}']) / len(its), 2) for f in ['BARE', 'JSON', 'TWICE']}, flush=True)
    with gzip.open(out_f, 'wt') as fh: json.dump({'model': key, 'res': res}, fh)

if __name__ == '__main__': main(*sys.argv[1:])
