"""Helper for one compression job of one model (PROTOCOL_ADDENDUM_9, post-smoke item 20): runs the locked doses in reverse
order with the same code path as the worker. The worker skips any dose whose result file already exists, so the two meet
in the middle; a dose both happen to run gives the same deterministic result and the later write replaces the earlier one.
  python3 gptq_helper.py <items_dir> <out_root> <model key> <damage> <name>
"""
import sys, os, json, torch, transformers
import atlas_comp  # noqa: F401  (patches atlas_full)
import atlas_full as A

items_dir, root, key, k, name = sys.argv[1:6]
mdir = f'{root}/{key}'; lock = json.load(open(f'{mdir}/lock_{k}.json'))['lock']
items, man = A.load_items(items_dir); tok, model, end = A.load_model(key); dm = A.Dmg(model); probe = A.Probe(model, tok, items['PROBE']['texts'])
meta = {'model_key': key, 'model': A.MODELS[key][0], 'items_sha': man, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'host': os.uname().nodename, 'worker': name, 'qbits': A.QBITS}
ctx = (model, tok, end, dm, items, probe, meta); A.restore_check(ctx)
for x in reversed(lock):
    c = {'damage': k, 'dose': x, 'seed': 0}
    if os.path.exists(A.cond_file(mdir, c)): print(name, 'skip', c, flush=True); continue
    A.run_cond(ctx, c, mdir)
A.restore_check(ctx); print(name, 'done', flush=True)
