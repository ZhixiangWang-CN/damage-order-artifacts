"""Exploratory pilot (not preregistered): instruction push on intact models.
  python3 push.py <items_dir> <model_key> <out.json.gz>
For every eval item, task and generated format (BARE, JSON, TWICE), score the canonical target output (target() + turn end) under
  with:    chat(body(it, fmt))            (the prompt the atlas used)
  without: chat(question only)            (format instruction removed)
and store both summed log-probs and the continuation length. Push = with - without (how far the instruction moves the model).
"""
import sys, os, gzip, json, torch
import atlas_full as A
import atlas_comp  # noqa: F401  (patches atlas_full: q32 loading)

def question(it): return it['q'] + ('\n' + '\n'.join(f"{'ABCDE'[i]}. {o}" for i, o in enumerate(it['opts'])) if it['kind'] == 'letter' else '')

@torch.no_grad()
def score(model, tok, pairs, bs=int(os.environ.get('PUSH_BS', 32))):   # batch size only (Gemma's 256k vocabulary needs a smaller one)
    """pairs: list of (prompt ids, continuation ids) -> summed log p of the continuation."""
    out = [None] * len(pairs); order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    for s in range(0, len(order), bs):
        b = order[s:s + bs]; seqs = [pairs[i][0] + pairs[i][1] for i in b]; L = max(map(len, seqs))
        x = torch.full((len(b), L), tok.pad_token_id); m = torch.zeros((len(b), L), dtype=torch.long)
        for r, q in enumerate(seqs): x[r, L - len(q):] = torch.tensor(q); m[r, L - len(q):] = 1
        pos = (m.cumsum(1) - 1).clamp(min=0)
        K = max(len(pairs[i][1]) for i in b); lg = model(input_ids=x.cuda(), attention_mask=m.cuda(), position_ids=pos.cuda()).logits[:, L - K - 1:L - 1]
        lp = lg.float().log_softmax(-1); del lg   # only the positions that predict continuation tokens (same values as the full log-softmax)
        for r, i in enumerate(b):
            c = pairs[i][1]; k = len(c); idx = torch.arange(K - k, K, device=lp.device)
            out[i] = float(lp[r, idx, torch.tensor(c, device=lp.device)].sum())
        del lp
    return out

def main(items_dir, key, out_f):
    items, _ = A.load_items(items_dir); tok, model, end = A.load_model(key); res = {}
    for t in A.TASKS:
        its = items[t]['eval']
        for f in ['BARE', 'JSON', 'TWICE']:
            cont = [tok.encode(A.target(it, f), add_special_tokens=False) + [end] for it in its]
            pw = [tok.encode(A.chat(tok, A.body(it, f)), add_special_tokens=False) for it in its]
            po = [tok.encode(A.chat(tok, question(it)), add_special_tokens=False) for it in its]
            w = score(model, tok, list(zip(pw, cont))); o = score(model, tok, list(zip(po, cont)))
            res[f'{t}_{f}'] = {'with': [round(v, 4) for v in w], 'without': [round(v, 4) for v in o], 'len': [len(c) for c in cont]}
        print(key, t, {f: round(sum(a - b for a, b in zip(res[f'{t}_{f}']['with'], res[f'{t}_{f}']['without'])) / len(its), 2) for f in ['BARE', 'JSON', 'TWICE']}, flush=True)
    with gzip.open(out_f, 'wt') as fh: json.dump({'model': key, 'res': res}, fh)

if __name__ == '__main__': main(*sys.argv[1:])
