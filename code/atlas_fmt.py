"""Format-compliance battery (PROTOCOL_ADDENDUM_6.md).
  python3 atlas_fmt.py smoke
  python3 atlas_fmt.py main <outdir> <shard> <nshards>
env: ATLAS_MODEL, ATLAS_NOISE_LOCK, ATLAS_QUANT_LOCK
"""
import sys, os, json, random, re, time, unicodedata, torch
import atlas, atlas_bench as B
from atlas import Damager, load, gen_items
CONTENT = ['FACT', 'ARITH1', 'COPY']; FMTS = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']
NOUN = {'FACT': 'name', 'ARITH1': 'number', 'COPY': 'word'}
INSTR = {'F0': 'Answer with just the {n}.',
         'F1': 'Begin your reply with "ANSWER:" followed by the {n}, and nothing else.',
         'F2': 'Reply with only a JSON object of the form {{"answer": "..."}} containing the {n}, and nothing else.',
         'F3': 'Reply with the {n} enclosed in double square brackets, like [[this]], and nothing else.',
         'F4': 'Reply with the {n}, then a space, then the word DONE, and nothing else.',
         'F5': 'Write the {n} twice, separated by a comma and a space, and nothing else.'}
FRE = {'F0': re.compile(r'^[^\n]{1,40}$'), 'F1': re.compile(r'^ANSWER:\s?\S[^\n]*$'), 'F2': re.compile(r'^\{\s*"answer"\s*:\s*("[^"\n]*"|-?\d+)\s*\}$'),
       'F3': re.compile(r'^\[\[[^\[\]\n]+\]\]$'), 'F4': re.compile(r'^[^\n]+ DONE\.?$'), 'F5': re.compile(r'^([^\n,]+), \1\.?$')}
def pseudo(r): return ''.join(r.choice('bcdfghjklmnprstvz') + r.choice('aeiou') for _ in range(3))
def items_for(task, n, seed):
    if task in ('FACT', 'ARITH1'): return gen_items(task, n, seed)
    r = random.Random(f'{task}-{seed}'); out = []
    for _ in range(n):
        if task == 'COPY': w = pseudo(r); out.append({'q': f'Repeat this word: {w}', 'ans': w})
        else: ws = ' '.join(pseudo(r) for _ in range(6)); out.append({'q': f'Repeat exactly the following six words, separated by single spaces, and nothing else: {ws}', 'ans': ws})
    return out
def deacc(x): return ''.join(c for c in unicodedata.normalize('NFKD', x) if not unicodedata.combining(c))
def content_ok(task, out, gold):
    o = deacc(out).lower()
    if task == 'ARITH1': return int(bool(re.search(r'(?<!\d)' + re.escape(gold) + r'(?!\d)', out)))
    if task == 'COPYLONG': return int(' '.join(o.split()).strip(' .') == gold.lower())
    return int(gold.lower() in o)
def format_ok(fmt, out): return int(bool(FRE[fmt].match(out.strip())))
@torch.no_grad()
def gen(model, tok, prompts, mx):
    outs = []
    for i in range(0, len(prompts), 40):
        enc = tok(prompts[i:i + 40], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        o = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    return outs
def chat(tok, body): return tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True)
def eval_fmt(model, tok, sets):
    res = {}
    for task in CONTENT:
        for f in FMTS:
            its = sets[task]; outs = gen(model, tok, [chat(tok, it['q'] + '\n' + INSTR[f].format(n=NOUN[task])) for it in its], 24)
            res[f'{task}_{f}'] = {'outputs': outs, 'content': [content_ok(task, o, it['ans']) for o, it in zip(outs, its)], 'format': [format_ok(f, o) for o in outs]}
    its = sets['COPYLONG']; outs = gen(model, tok, [chat(tok, it['q']) for it in its], 40)
    res['COPYLONG'] = {'outputs': outs, 'content': [content_ok('COPYLONG', o, it['ans']) for o, it in zip(outs, its)], 'format': [1] * len(outs)}
    return res
def conds():
    nz = json.load(open(os.environ.get('ATLAS_NOISE_LOCK', 'LOCKED_DOSES.json')))['noise']; qz = json.load(open(os.environ.get('ATLAS_QUANT_LOCK', 'LOCKED_DOSES.json')))['quant']
    return [{'damage': 'none', 'dose': 0}] + [{'damage': 'noise', 'dose': x, 'seed': s} for x in nz for s in (1, 2, 3)] + [{'damage': 'quant', 'dose': q} for q in qz] + [{'damage': 'ld_rand', 'dose': n, 'seed': s} for n in (1, 2, 3, 4, 5, 6, 8, 10) for s in (1, 2, 3)]
if __name__ == '__main__':
    mode = sys.argv[1]; tok, model = load()
    for t in list(B.FORMATS):
        if t in B.OLD: B.FORMATS[t] = ['GEN', 'CHOICE']
        else: del B.FORMATS[t]
    if mode == 'smoke':
        sets = {t: items_for(t, 40, 'smoke') for t in CONTENT + ['COPYLONG']}; r = eval_fmt(model, tok, sets)
        print('FORMAT', {k: round(sum(v['format']) / 40, 2) for k, v in r.items()}); print('CONTENT', {k: round(sum(v['content']) / 40, 2) for k, v in r.items()})
        for k in r:
            bad = [o for o, f, c in zip(r[k]['outputs'], r[k]['format'], r[k]['content']) if not (f and c)][:2]
            print(k, 'ok example:', repr(r[k]['outputs'][0]), '| failures:', [repr(x) for x in bad])
    else:
        out = sys.argv[2]; os.makedirs(out, exist_ok=True); dm = Damager(model); sh, ns = int(sys.argv[3]), int(sys.argv[4])
        sets = {t: items_for(t, 200, 'eval') for t in CONTENT + ['COPYLONG']}; ref = {t: (B.get_items(t, 200, 'eval'), 'eval') for t in B.FORMATS}
        for i, c in enumerate(conds()):
            if i % ns != sh: continue
            f = f"{out}/fmt_{c['damage']}_{c['dose']}_{c.get('seed', 0)}.json"
            if os.path.exists(f): continue
            t = time.time(); dm.apply(c); r = eval_fmt(model, tok, sets); rf = B.evaluate(model, tok, ref)
            json.dump({'cond': c, 'model': atlas.MODEL, 'fmt': r, 'ref': rf}, open(f, 'w'), ensure_ascii=False)
            print('fmt', c, {k: round(sum(v['format']) / len(v['format']), 2) for k, v in r.items()}, round(time.time() - t), 's', flush=True)
