"""Per-token decision margins of the intact model (PROTOCOL_ADDENDUM_10). No damage is applied.
  python3 margins.py <items_dir> <v8_model_dir> <out_file> <model key>
v8_model_dir holds the model's addendum-8 intact result (none_0_0.json.gz); for q32 it is the addendum-9 directory.
Generation repeats atlas_full.evaluate exactly (same prompts, greedy, same generation config and batching rule), with the
processed scores of every step kept: the gaps from the chosen token to the next four are what decided each greedy step.
"""
import sys, gzip, json
import numpy as np, torch
import atlas_comp  # noqa: F401  (patches atlas_full: q32 loading)
import atlas_full as A

GEN = ['BARE', 'JSON', 'TWICE']

@torch.no_grad()
def generate_gaps(model, tok, prompts, mx):
    """greedy generation as atlas_full.generate; returns per prompt (token ids through the first stop token, top-5 gaps per step)."""
    enc = [tok.encode(p, add_special_tokens=False) for p in prompts]; out = [None] * len(prompts)
    stop = model.generation_config.eos_token_id; stop = set(stop if isinstance(stop, list) else [stop])
    for b in A.batches([len(e) for e in enc], mx):
        e = tok([prompts[i] for i in b], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        o = model.generate(**e, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
        seq = o.sequences[:, e['input_ids'].shape[1]:].cpu()
        top = torch.stack([s.float().topk(5, dim=-1).values for s in o.scores], 1).cpu()   # batch x steps x 5
        for j, i in enumerate(b):
            ids = seq[j].tolist(); n = next((k + 1 for k, t in enumerate(ids) if t in stop), len(ids))
            out[i] = (ids[:n], (top[j, :n, :1] - top[j, :n, 1:]).numpy().round(4).tolist())
        del o, top
    return out

def crit_steps(tok, it, fmt, ids, correct):
    """number of steps that decide correctness (protocol: BARE up to the shortest prefix judged correct; JSON, TWICE all steps)."""
    if fmt != 'BARE' or not correct: return len(ids)
    for k in range(1, len(ids) + 1):
        if A.judge(it, 'BARE', tok.decode(ids[:k], skip_special_tokens=True))[0]: return k
    return len(ids)

def main(items_dir, v8dir, out_file, key):
    items, _ = A.load_items(items_dir); tok, model, end = A.load_model(key)
    with gzip.open(f'{v8dir}/none_0_0.json.gz', 'rt') as fh: v8 = json.load(fh)['res']
    jobs = [(t, i, it) for t in A.TASKS for i, it in enumerate(items[t]['eval'])]; res = {}
    for f in GEN:
        g = generate_gaps(model, tok, [A.chat(tok, A.body(it, f)) for t, i, it in jobs], A.GEN_MX[f]); k = 0
        for t in A.TASKS:
            its = items[t]['eval']; rows = g[k:k + len(its)]; k += len(its)
            txt = [tok.decode(ids, skip_special_tokens=True) for ids, _ in rows]; cor = [A.judge(it, f, s)[0] for it, s in zip(its, txt)]
            res[f'{t}_{f}'] = {'correct': cor, 'same_as_v8': [int(s == o) for s, o in zip(txt, v8[f'{t}_{f}']['outputs'])],
                               'crit': [crit_steps(tok, it, f, ids, c) for it, (ids, _), c in zip(its, rows, cor)],
                               'gaps': [gp for _, gp in rows], 'ids': [ids for ids, _ in rows]}
        print(key, f, 'same as v8', round(float(np.mean([x for t in A.TASKS for x in res[f'{t}_{f}']['same_as_v8']])), 4), flush=True)
    for t in A.TASKS:   # CHOICE: scores and correctness from the addendum-8 intact result; token counts of each candidate + turn end
        its = items[t]['eval']
        res[f'{t}_CHOICE'] = {'correct': v8[f'{t}_CHOICE']['correct'], 'scores': v8[f'{t}_CHOICE']['scores'],
                              'lens': [[len(tok.encode(c, add_special_tokens=False)) + 1 for c in it['cands']] for it in its]}
    meta = {'model_key': key, 'torch': torch.__version__, 'gen_config': model.generation_config.to_dict()}
    with gzip.open(out_file + '.tmp', 'wt') as fh: json.dump({'meta': meta, 'res': res}, fh)
    import os; os.replace(out_file + '.tmp', out_file); print(key, 'done', flush=True)

if __name__ == '__main__': main(*sys.argv[1:5])
