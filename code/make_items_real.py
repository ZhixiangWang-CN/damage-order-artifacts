"""Build and freeze the four realistic task sets of PROTOCOL_ADDENDUM_11 (run once, on server_b, which can reach the Hugging Face hub).
  python3 make_items_real.py <outdir>
Writes <outdir>/<TASK>.json = {'eval': [...200], 'smoke': [...10]} for GSM_COT, CODE, IFEVAL, FCALL, and MANIFEST.json (sha256).
"""
import sys, os, json, random, re, hashlib
from datasets import load_dataset
from huggingface_hub import hf_hub_download

N = 200; N_SMOKE = 10

def gsm_cot():   # same source, shuffle and filter as make_items.gsm8k_cot (whose 100 items are the first 100 here)
    d = list(load_dataset('openai/gsm8k', 'main', split='test')); random.Random('GSM8K-eval').shuffle(d); out = []
    for x in d:
        a = x['answer'].split('####')[-1].strip().replace(',', '')
        if re.fullmatch(r'-?\d+', a): out.append({'q': x['question'].strip(), 'ans': a})
        if len(out) == N + N_SMOKE: break
    return {'eval': out[:N], 'smoke': out[N:]}

def code():   # all 164 HumanEval problems + MBPP (sanitized, test split) to 200; 10 more MBPP for smoke
    he = [{'src': 'humaneval', 'id': x['task_id'], 'prompt': x['prompt'], 'test': x['test'] + f"\ncheck({x['entry_point']})\n", 'entry': x['entry_point']}
          for x in load_dataset('openai/openai_humaneval', split='test')]
    mb = list(load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')); random.Random('MBPP').shuffle(mb)
    mb = [{'src': 'mbpp', 'id': str(x['task_id']), 'prompt': x['prompt'].strip(), 'tests': x['test_list'],
           'test': '\n'.join(x['test_imports']) + '\n' + '\n'.join(x['test_list']) + '\n'} for x in mb]
    return {'eval': he + mb[:N - len(he)], 'smoke': mb[N - len(he):N - len(he) + N_SMOKE]}

LONG = {'length_constraints:number_words': lambda kw: kw.get('relation') == 'at least' and (kw.get('num_words') or 0) > 200,
        'length_constraints:number_sentences': lambda kw: kw.get('relation') == 'at least' and (kw.get('num_sentences') or 0) > 12,
        'length_constraints:number_paragraphs': lambda kw: (kw.get('num_paragraphs') or 0) > 4,
        'length_constraints:nth_paragraph_first_word': lambda kw: (kw.get('num_paragraphs') or 0) > 4}
def ifeval():   # prompts whose constraints can be met within 384 new tokens (no long-length demands), then 200 at random
    d = list(load_dataset('google/IFEval', split='train')); keep = []
    for x in d:
        kws = [{k: v for k, v in kw.items() if v is not None} for kw in x['kwargs']]
        if any(i in LONG and LONG[i](kw) for i, kw in zip(x['instruction_id_list'], kws)): continue
        keep.append({'key': x['key'], 'prompt': x['prompt'], 'ids': x['instruction_id_list'], 'kwargs': kws})
    random.Random('IFEVAL').shuffle(keep); return {'eval': keep[:N], 'smoke': keep[N:N + N_SMOKE], 'n_pool': len(keep), 'n_source': len(d)}

def fcall():   # BFCL v3 "simple": one function, arguments checked against the possible answers
    rd = lambda f: [json.loads(l) for l in open(hf_hub_download('gorilla-llm/Berkeley-Function-Calling-Leaderboard', f, repo_type='dataset')) if l.strip()]
    q = rd('BFCL_v3_simple.json'); a = {x['id']: x['ground_truth'] for x in rd('possible_answer/BFCL_v3_simple.json')}
    out = [{'id': x['id'], 'question': x['question'][0][-1]['content'], 'function': x['function'][0], 'gt': a[x['id']][0]} for x in q if x['id'] in a]
    random.Random('FCALL').shuffle(out); return {'eval': out[:N], 'smoke': out[N:N + N_SMOKE]}

if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); man = {}
    for t, f in (('GSM_COT', gsm_cot), ('CODE', code), ('IFEVAL', ifeval), ('FCALL', fcall)):
        x = f(); s = json.dumps(x, ensure_ascii=False, sort_keys=True); open(f'{out}/{t}.json', 'w').write(s); man[t] = hashlib.sha256(s.encode()).hexdigest()
        print(t, {k: (len(v) if isinstance(v, list) else v) for k, v in x.items()}, man[t][:12], flush=True)
    json.dump(man, open(f'{out}/MANIFEST.json', 'w'), indent=1)
