"""Output check (PROTOCOL_ADDENDUM_5.md): re-run generative formats and save raw outputs. python3 atlas_out.py <outdir>"""
import sys, os, json, time, torch
import atlas, atlas_bench as B
from atlas import Damager, load
TASKS = {'ARC': 'GENOPT', 'CSQA': 'GENOPT', 'MMLU_CALC': 'GENOPT', 'MMLU_MEM': 'GENOPT', 'SCIQ': 'GEN', 'BOOLQ': 'GEN', 'ARITH3': 'GEN', 'FACT': 'GEN', 'GSM8K': 'COT'}
CONDS = [{'damage': 'none', 'dose': 0}] + [{'damage': 'noise', 'dose': x, 'seed': s} for x in (0.20, 0.31, 0.375) for s in (1, 2)] + [{'damage': 'quant', 'dose': q} for q in (10, 7, 6)] + [{'damage': 'ld_rand', 'dose': n, 'seed': 1} for n in (2, 4, 6)]
@torch.no_grad()
def gen_all(model, tok, sets):
    res = {}
    for task, fmt in TASKS.items():
        items = sets[task][0][:100 if fmt == 'COT' else 200]; ps, gold = B.prompts_for(tok, task, items, fmt, sets[task][1]); outs = []
        bs = 12 if (fmt == 'COT' or task == 'BOOLQ') else 32; mx = 384 if fmt == 'COT' else (24 if fmt == 'GENOPT' else 12)
        for i in range(0, len(ps), bs):
            enc = tok(ps[i:i + bs], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        res[f'{task}_{fmt}'] = {'outputs': outs, 'gold': gold, 'correct': [B.judge(task, fmt, it, t, g) for it, t, g in zip(items, outs, gold)]}
    return res
if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); tok, model = load(); dm = Damager(model)
    sets = {t: (B.get_items(t, 200, 'eval'), 'eval') for t in TASKS}
    for c in CONDS:
        f = f"{out}/out_{c['damage']}_{c['dose']}_{c.get('seed', 0)}.json"
        if os.path.exists(f): continue
        t = time.time(); dm.apply(c); r = gen_all(model, tok, sets); json.dump({'cond': c, 'eval': r}, open(f, 'w'), ensure_ascii=False)
        print('out', c, {k: round(sum(v['correct']) / len(v['correct']), 2) for k, v in r.items()}, round(time.time() - t), 's', flush=True)
