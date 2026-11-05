"""Build and freeze all item sets for PROTOCOL_ADDENDUM_8 (run once, on server_a where the datasets are cached).
  HF_DATASETS_OFFLINE=1 python3 make_items.py <outdir>
Writes <outdir>/<TASK>.json = {'eval': [...300], 'calib': [...40], 'smoke': [...30]}, GSM8K_COT.json, PROBE.json, MANIFEST.json (sha256).
"""
import sys, os, json, random, re, glob, hashlib
import pandas as pd
from atlas import gen_items
from atlas_fmt import items_for, pseudo

N_EVAL, N_CAL, N_SMOKE = 300, 40, 30
LET = 'ABCDE'
SHORT = ['ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'COPY', 'SCIQ', 'BOOLQ']
OPTION = ['ARC', 'CSQA', 'MMLU_CALC', 'MMLU_MEM', 'HELLASWAG', 'PIQA', 'WINOGRANDE', 'OBQA']
TASKS = SHORT + OPTION

def parquet(repo_dir, sub, split):
    roots = [os.path.expanduser('~/.cache/huggingface/hub'), os.path.join(os.environ.get('HF_HOME', '/nonexistent'), 'hub')]
    for r in roots:
        fs = sorted(glob.glob(f'{r}/{repo_dir}/snapshots/*/{sub}/{split}-*.parquet'))
        if fs: return pd.concat([pd.read_parquet(f) for f in fs]).to_dict('records')
    raise FileNotFoundError(f'{repo_dir}/{sub}/{split}')

def letter_item(q, opts, gold_idx, r):
    order = list(range(len(opts))); r.shuffle(order)
    o = [str(opts[i]).strip() for i in order]; g = order.index(gold_idx)
    return {'q': q.strip(), 'opts': o, 'ans': LET[g], 'cands': [LET[g]] + [LET[i] for i in range(len(o)) if i != g], 'kind': 'letter'}

def synthetic(task):
    out = {}
    for split, n in (('eval', N_EVAL), ('calib', N_CAL), ('smoke', N_SMOKE)):
        if task == 'COPY':
            its = items_for('COPY', n, split); r = random.Random(f'COPYDIST-{split}')
            for it in its:
                w = it['ans']; ds = set()
                while len(ds) < 3:
                    k = r.randrange(3); v = w[:2 * k] + r.choice('bcdfghjklmnprstvz') + r.choice('aeiou') + w[2 * k + 2:]
                    if v != w: ds.add(v)
                it.update({'cands': [w] + sorted(ds), 'kind': 'word'})
        else:
            its = gen_items(task, n, split)
            for it in its:
                it['cands'] = [it['ans']] + it.pop('dist')
                if task == 'FACT': it['kind'] = 'name'
        out[split] = its
    return out

def from_pool(pool):
    assert len(pool) >= N_EVAL + N_CAL + N_SMOKE, len(pool)
    return {'eval': pool[:N_EVAL], 'calib': pool[N_EVAL:N_EVAL + N_CAL], 'smoke': pool[N_EVAL + N_CAL:N_EVAL + N_CAL + N_SMOKE]}

def bench(task):
    from datasets import load_dataset
    r = random.Random(f'{task}-eval'); ro = random.Random(f'opts-{task}'); pool = []
    if task == 'SCIQ':   # same filter and shuffle as atlas_bench.bench_items (first 200 identical to addendum 4)
        d = list(load_dataset('allenai/sciq', split='test')); r.shuffle(d)
        for x in d:
            a = x['correct_answer'].strip()
            if len(a.split()) > 3: continue
            pool.append({'q': x['question'].strip(), 'ans': a, 'cands': [a, x['distractor1'].strip(), x['distractor2'].strip(), x['distractor3'].strip()], 'kind': 'text'})
    elif task == 'BOOLQ':
        d = list(load_dataset('google/boolq', split='validation')); r.shuffle(d)
        for x in d:
            if len(x['passage'].split()) > 150: continue
            a = 'yes' if x['answer'] else 'no'
            pool.append({'q': f"Passage: {x['passage'].strip()}\nQuestion: {x['question'].strip()}?", 'ans': a, 'cands': [a, 'no' if a == 'yes' else 'yes'], 'kind': 'yesno'})
    elif task in ('ARC', 'CSQA'):
        d = list(load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test') if task == 'ARC' else load_dataset('tau/commonsense_qa', split='validation')); r.shuffle(d)
        k = 4 if task == 'ARC' else 5
        for x in d:
            if len(x['choices']['text']) != k or x['answerKey'] not in x['choices']['label']: continue
            pool.append(letter_item(x['question'], x['choices']['text'], x['choices']['label'].index(x['answerKey']), ro))
    elif task.startswith('MMLU'):
        subs = ['high_school_mathematics', 'elementary_mathematics', 'high_school_physics', 'college_physics'] if task == 'MMLU_CALC' else ['high_school_world_history', 'nutrition', 'high_school_geography', 'high_school_psychology']
        d = [x for x in load_dataset('cais/mmlu', 'all', split='test') if x['subject'] in subs and len(x['question'].split()) <= 180]; r.shuffle(d)
        for x in d: pool.append(letter_item(x['question'], x['choices'], int(x['answer']), ro))
    elif task == 'HELLASWAG':
        d = parquet('datasets--Rowan--hellaswag', 'data', 'validation'); r.shuffle(d)
        for x in d:
            if not str(x['label']).strip().isdigit(): continue
            pool.append(letter_item(f"Choose the most plausible continuation of the text.\nText: {x['ctx'].strip()}", list(x['endings']), int(x['label']), ro))
    elif task == 'PIQA':
        d = parquet('datasets--lighteval--piqa', 'plain_text', 'validation'); r.shuffle(d)
        for x in d: pool.append(letter_item(f"Which solution better achieves the goal?\nGoal: {x['goal'].strip()}", [x['sol1'], x['sol2']], int(x['label']), ro))
    elif task == 'WINOGRANDE':
        d = parquet('datasets--winogrande', 'winogrande_xl', 'validation'); r.shuffle(d)
        for x in d: pool.append(letter_item(f"Which option correctly fills the blank (_) in the sentence?\nSentence: {x['sentence'].strip()}", [x['option1'], x['option2']], int(x['answer']) - 1, ro))
    elif task == 'OBQA':
        d = parquet('datasets--allenai--openbookqa', 'main', 'test') + parquet('datasets--allenai--openbookqa', 'main', 'validation'); r.shuffle(d)
        for x in d:
            ch = x['choices']; labels = list(ch['label'])
            if x['answerKey'] not in labels: continue
            pool.append(letter_item(x['question_stem'], list(ch['text']), labels.index(x['answerKey']), ro))
    return from_pool(pool)

def gsm8k_cot():
    from datasets import load_dataset
    d = list(load_dataset('openai/gsm8k', 'main', split='test')); random.Random('GSM8K-eval').shuffle(d); out = []
    for x in d:
        a = x['answer'].split('####')[-1].strip().replace(',', '')
        if re.fullmatch(r'-?\d+', a): out.append({'q': x['question'].strip(), 'ans': a, 'kind': 'number'})
        if len(out) == 100: break
    return {'eval': out, 'calib': [], 'smoke': out[:10]}

def probe():
    from datasets import load_dataset
    d = [x['support'].strip() for x in load_dataset('allenai/sciq', split='test') if len(x['support'].split()) >= 60]
    random.Random('probe').shuffle(d); return {'texts': d[:64]}

if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); man = {}
    for t in TASKS + ['GSM8K_COT', 'PROBE']:
        x = synthetic(t) if t in ('ARITH1', 'ARITH2', 'ARITH3', 'FACT', 'RETR', 'COPY') else gsm8k_cot() if t == 'GSM8K_COT' else probe() if t == 'PROBE' else bench(t)
        s = json.dumps(x, ensure_ascii=False, sort_keys=True); open(f'{out}/{t}.json', 'w').write(s); man[t] = hashlib.sha256(s.encode()).hexdigest()
        print(t, {k: len(v) for k, v in x.items()}, man[t][:12], flush=True)
    json.dump(man, open(f'{out}/MANIFEST.json', 'w'), indent=1)
