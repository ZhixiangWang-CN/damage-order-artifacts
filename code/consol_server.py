"""Server-side helper for CONSOLIDATION_PLAN: token counts of target outputs and GSM8K step groups."""
import json, re, random, glob
import numpy as np
from transformers import AutoTokenizer
import atlas_fmt as F, atlas_bench as B
from atlas import gen_items
out = {}
sets = {t: F.items_for(t, 200, 'eval') for t in F.CONTENT + ['COPYLONG']}; ref = {t: B.get_items(t, 200, 'eval') for t in B.OLD}
def target(task, f, a):
    return {'F0': a, 'F1': f'ANSWER: {a}', 'F2': '{"answer": "%s"}' % a, 'F3': f'[[{a}]]', 'F4': f'{a} DONE', 'F5': f'{a}, {a}'}[f]
for name, mid in [('qwen', 'Qwen/Qwen2.5-7B-Instruct'), ('mistral', 'mistralai/Mistral-7B-Instruct-v0.3')]:
    tok = AutoTokenizer.from_pretrained(mid); n = lambda s: len(tok.encode(s, add_special_tokens=False)); T = {}
    for t in F.CONTENT:
        for f in F.FMTS: T[f'{t}_{f}'] = float(np.mean([n(target(t, f, it['ans'])) for it in sets[t]]))
    T['COPYLONG'] = float(np.mean([n(it['ans']) for it in sets['COPYLONG']]))
    for t in B.OLD: T[f'{t}_GEN'] = float(np.mean([n(it['ans']) for it in ref[t]])); T[f'{t}_CHOICE'] = 1.0
    out[f'T_{name}'] = T
g = B.get_items('GSM8K', 200, 'eval')
from datasets import load_dataset
d = {x['question'].strip(): x['answer'].count('<<') for x in load_dataset('openai/gsm8k', 'main', split='test')}
steps = [d[it['q']] for it in g]; out['gsm8k_steps'] = steps
R = {}
for f in glob.glob('bench/bench_*.json'):
    x = json.load(open(f)); c = x['cond']
    if c['damage'] in ('none', 'noise', 'ld_rand'): R[(c['damage'], c['dose'], c.get('seed', 0))] = {m: x['eval'][m]['correct'] for m in ('GSM8K_CHOICE', 'GSM8K_COT')}
out['gsm8k'] = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in R.items()}
json.dump(out, open('consol_server.json', 'w'))
print({k: {m: round(v, 1) for m, v in out[k].items()} for k in ('T_qwen', 'T_mistral')}); print('steps hist', {s: steps.count(s) for s in sorted(set(steps))})
