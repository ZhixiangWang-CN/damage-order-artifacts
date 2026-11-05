"""Runner for PROTOCOL_ADDENDUM_11: four realistic tasks (GSM_COT, CODE, IFEVAL, FCALL) under the locked noise and Wanda doses.
Damage, doses, seeds and the probe are those of atlas_full.py / atlas_comp.py (imported), so a condition reproduces the weights
of the addendum-8/9 condition with the same damage, dose and seed; the probe is compared with the archived one as a check.
  python3 atlas_real.py worker <items_real> <items_v8> <out_root> <model keys> <damages> <ref_dir_noise> <ref_dir_wanda> <name>
  python3 atlas_real.py smoke  <items_real> <out_dir> <model key>          (intact model, smoke items, judges printed)
<ref_dir_*> hold the archived conditions of each model (<ref_dir>/<key>/<damage>_<dose>_<seed>.json.gz and lock_<damage>.json).
env: ATLAS_TOK_BUDGET, ATLAS_C4 (as in atlas_full.py / atlas_comp.py); REAL_SEEDS (noise seeds this worker may take), REAL_NO_INTACT (skip the intact condition)
"""
import sys, os, re, json, gzip, time, shutil, traceback, tempfile, subprocess, gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, transformers
import atlas_comp  # noqa: F401  (patches atlas_full: compression damages, multi-GPU loading)
import atlas_full as A

TASKS = ['GSM_COT', 'CODE', 'IFEVAL', 'FCALL']; MX = {'GSM_COT': 384, 'CODE': 384, 'IFEVAL': 384, 'FCALL': 128}
SEEDS = {'noise': (1, 2), 'wanda': (0,)}; PROBE_TOL = 1e-3

# ---------------------------------------------------------------- prompts
def prompt(t, it):
    if t == 'GSM_COT': return it['q'] + '\nThink step by step, briefly, then give the final answer in the form "The answer is N."'
    if t == 'CODE':
        if it['src'] == 'humaneval':
            return ('Complete the following Python function. Respond with the complete function, including its signature, in a single Python code block and nothing else.\n\n'
                    f"```python\n{it['prompt']}```")
        return f"{it['prompt']}\nYour code should pass this test:\n{it['tests'][0]}\nRespond with the Python function in a single Python code block and nothing else."
    if t == 'IFEVAL': return it['prompt']
    return (f"You can call the following function:\n{json.dumps(it['function'], indent=1)}\n\nRequest: {it['question']}\n\n"
            'Respond with only a JSON object of the form {"name": "<function name>", "arguments": {"<argument name>": <value>, ...}} and nothing else.')

# ---------------------------------------------------------------- judges
def code_block(s):
    m = re.search(r'```(?:python|py)?\s*\n(.*?)```', s, re.S) or re.search(r'```(?:python|py)?\s*\n(.*)', s, re.S)
    return m.group(1) if m else s
GUARD = '''import builtins, os, shutil, subprocess, resource, faulthandler
try: resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))   # Linux; macOS (local tests) refuses it
except ValueError: pass
faulthandler.disable()
for _m, _n in ((os, 'system'), (os, 'kill'), (os, 'putenv'), (os, 'remove'), (os, 'removedirs'), (os, 'rmdir'), (os, 'fchdir'), (os, 'setuid'), (os, 'fork'),
               (os, 'forkpty'), (os, 'killpg'), (os, 'rename'), (os, 'renames'), (os, 'truncate'), (os, 'replace'), (os, 'unlink'), (os, 'fchmod'), (os, 'fchown'),
               (os, 'chmod'), (os, 'chown'), (os, 'chroot'), (os, 'lchown'), (shutil, 'rmtree'), (shutil, 'move'), (shutil, 'chown'), (subprocess, 'Popen')):
    setattr(_m, _n, None)
'''
def run_code(program, timeout=10):
    """run one program in an isolated interpreter (fresh temp dir, empty environment, 2 GB address space, destructive calls disabled)."""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'p.py'); open(f, 'w').write(GUARD + program)
        try: r = subprocess.run([sys.executable, '-I', f], cwd=d, env={'PATH': '/usr/bin:/bin'}, capture_output=True, timeout=timeout); return int(r.returncode == 0)
        except subprocess.TimeoutExpired: return 0
def judge_code(it, out):
    c = code_block(out)
    prog = (it['prompt'] + '\n    pass\n\n' + c if it['src'] == 'humaneval' else c) + '\n\n' + it['test']
    return run_code(prog)

_IF = None
def judge_ifeval(it, out):
    global _IF
    if _IF is None:
        from ifeval_lib import instructions_registry as R; _IF = R
    if not out.strip(): return 0
    for iid, kw in zip(it['ids'], it['kwargs']):
        inst = _IF.INSTRUCTION_DICT[iid](iid); inst.build_description(**kw)
        if 'prompt' in (inst.get_instruction_args() or {}): inst.build_description(prompt=it['prompt'])
        if not inst.check_following(out): return 0
    return 1

def _std(v):   # BFCL-style standardisation of strings
    return re.sub(r"[ ,./\-_*^]", '', str(v)).lower().replace("'", '"')
def _eq(x, a):
    if isinstance(a, bool) or isinstance(x, bool): return isinstance(x, bool) and isinstance(a, bool) and x == a
    if isinstance(a, (int, float)): return isinstance(x, (int, float)) and float(x) == float(a)
    if isinstance(a, str): return isinstance(x, str) and _std(x) == _std(a)
    if isinstance(a, list): return isinstance(x, list) and len(x) == len(a) and all(_eq(p, q) for p, q in zip(x, a))
    if isinstance(a, dict): return isinstance(x, dict) and set(x) <= set(a) and all(any(_eq(x[k], q) for q in a[k]) for k in x) and all('' in a[k] for k in a if k not in x)
    return x == a
def judge_fcall(it, out):
    """(correct, format ok, content anywhere). format: the whole reply (optionally inside one code fence) is a JSON object with keys
    name and arguments, arguments an object whose keys are parameters of the function. content: name and every argument match the
    possible answers (a parameter whose answers include "" may be omitted). content anywhere: the function name and one acceptable
    value of every required argument occur in the reply as text."""
    fn = it['function']; name, gt = next(iter(it['gt'].items())); props = fn['parameters'].get('properties', {})
    s = out.strip(); m = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', s, re.S); s = m.group(1).strip() if m else s
    try: o = json.loads(s)
    except Exception: o = None
    fok = isinstance(o, dict) and set(o) == {'name', 'arguments'} and isinstance(o['name'], str) and isinstance(o['arguments'], dict) and set(o['arguments']) <= set(props)
    cok = fok and o['name'] == name and all(any(_eq(o['arguments'][k], a) for a in gt[k]) if k in o['arguments'] else '' in gt[k] for k in gt)
    t = _std(out)
    def forms(a):   # a number may be written as 3 or 3.0
        if isinstance(a, (int, float)) and not isinstance(a, bool) and float(a).is_integer(): return {_std(int(a)), _std(float(a))}
        return {_std(json.dumps(a) if isinstance(a, dict) else a)}
    def appears(a): return all(appears(p) for p in a) if isinstance(a, list) else any(f in t for f in forms(a))
    anyw = _std(name) in t and all(any(appears(a) for a in gt[k] if a != '') for k in gt if '' not in gt[k])
    return int(fok and cok), int(fok), int(anyw)

def judge_all(t, its, outs):
    if t == 'GSM_COT': return {'correct': [A.judge_cot(o, it['ans']) for it, o in zip(its, outs)]}
    if t == 'IFEVAL': return {'correct': [judge_ifeval(it, o) for it, o in zip(its, outs)]}
    if t == 'CODE':
        with ThreadPoolExecutor(8) as ex: return {'correct': list(ex.map(judge_code, its, outs))}
    J = [judge_fcall(it, o) for it, o in zip(its, outs)]
    return {'correct': [j[0] for j in J], 'format_ok': [j[1] for j in J], 'content_any': [j[2] for j in J]}

def evaluate(model, tok, items, split):
    res = {}
    for t in TASKS:
        its = items[t][split]; outs = A.generate(model, tok, [A.chat(tok, prompt(t, it)) for it in its], MX[t])
        r = judge_all(t, its, outs); r['outputs'] = outs; r['chance'] = 0.0; r['acc'] = float(np.mean(r['correct'])); res[t] = r
    return res

# ---------------------------------------------------------------- worker (one claim per condition)
def load_real(d): return {t: json.load(open(f'{d}/{t}.json')) for t in TASKS}, json.load(open(f'{d}/MANIFEST.json'))
def conds(key, damages, refs):
    only = os.environ.get('REAL_SEEDS')   # split one model's noise seeds between machines (claims are per machine); 'none' runs only where the intact is wanted
    cs = [] if os.environ.get('REAL_NO_INTACT') else [(key, 'none', 0, 0)]
    for k in damages:
        for x in json.load(open(f'{refs[k]}/{key}/lock_{k}.json'))['lock']:
            for s in SEEDS[k]:
                if not only or k != 'noise' or str(s) in only.split(','): cs.append((key, k, x, s))
    return cs
def worker(items_real, items_v8, root, keys, damages, refs, name):
    items, man = load_real(items_real); v8items, _ = A.load_items(items_v8); os.makedirs(f'{root}/claims', exist_ok=True)
    js = [c for k in keys for c in conds(k, damages, refs)]; cur = None; ctx = None; nfail = 0
    while True:
        todo = [c for c in js if not os.path.exists(f'{root}/claims/{"__".join(map(str, c))}')]
        if not todo: print(name, 'no jobs left', flush=True); return
        todo.sort(key=lambda c: (c[0] != cur, c[1] != 'none')); c = todo[0]; cl = f'{root}/claims/{"__".join(map(str, c))}'
        try: os.makedirs(cl)
        except FileExistsError: continue
        key, k, x, s = c; mdir = f'{root}/{key}'; os.makedirs(mdir, exist_ok=True); f = f'{mdir}/{k}_{x}_{s}.json.gz'
        try:
            if key != cur:
                ctx = None; gc.collect(); torch.cuda.empty_cache(); tok, model, end = A.load_model(key); dm = A.Dmg(model)
                probe = A.Probe(model, tok, v8items['PROBE']['texts']); ctx = (tok, model, dm, probe); cur = key
            tok, model, dm, probe = ctx; dm.restore(); t0 = time.time()
            if k != 'none': dm.apply({'damage': k, 'dose': x, 'seed': s})
            pr = probe.measure(model); ref = None
            if k != 'none':
                with gzip.open(f'{refs[k]}/{key}/{k}_{x}_{s}.json.gz', 'rt') as fh: ref = json.load(fh)['probe']
            ok = ref is None or abs(pr['nll'] - ref['nll']) <= PROBE_TOL * max(1.0, abs(ref['nll']))
            ev = evaluate(model, tok, items, 'eval')
            out = {'cond': {'damage': k, 'dose': x, 'seed': s}, 'probe': pr, 'probe_ref': ref, 'check_ok': bool(ok), 'res': ev, 'secs': round(time.time() - t0),
                   'model_key': key, 'items_sha': man, 'torch': torch.__version__, 'transformers': transformers.__version__, 'host': os.uname().nodename, 'worker': name}
            with gzip.open(f + '.tmp', 'wt') as fh: json.dump(out, fh, ensure_ascii=False)
            os.replace(f + '.tmp', f); open(f'{cl}/done', 'w').write(name); nfail = 0
            print(name, c, 'check', ok, 'probe', round(pr['nll'], 4), (ref or {}).get('nll'), {t: round(v['acc'], 3) for t, v in ev.items()}, out['secs'], 's', flush=True)
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True); nfail += 1; ctx = None; cur = None; gc.collect(); torch.cuda.empty_cache()
            if 'out of memory' in tb: shutil.rmtree(cl, ignore_errors=True); time.sleep(300)
            else: open(f'{cl}/FAILED', 'w').write(tb)
            if nfail >= 3: print(name, 'three failures in a row, exiting', flush=True); return

def smoke(items_real, out_dir, key):
    items, _ = load_real(items_real); tok, model, end = A.load_model(key); ev = evaluate(model, tok, items, 'smoke'); os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(f'{out_dir}/smoke_real_{key}.json', 'w'), indent=1, ensure_ascii=False)
    for t, v in ev.items(): print(t, 'acc', round(v['acc'], 2), {k: v[k] for k in ('format_ok', 'content_any') if k in v})

if __name__ == '__main__':
    if sys.argv[1] == 'smoke': smoke(*sys.argv[2:5])
    else:
        a = sys.argv[2:]; worker(a[0], a[1], a[2], a[3].split(','), a[4].split(','), {'noise': a[5], 'wanda': a[6]}, a[7])
