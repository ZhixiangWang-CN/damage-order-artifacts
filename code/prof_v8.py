"""time each part of one smoke evaluation.  python3 prof_v8.py <model_key> <items_dir>"""
import sys, time, torch, atlas_full as A
items, _ = A.load_items(sys.argv[2]); tok, model, end = A.load_model(sys.argv[1])
for f in ['CHOICE', 'BARE', 'JSON', 'TWICE']:
    torch.cuda.synchronize(); t = time.time(); A.evaluate(model, tok, end, items, 'smoke', formats=[f]); torch.cuda.synchronize(); print(f, round(time.time() - t, 1), 's', flush=True)
t = time.time(); A.evaluate(model, tok, end, items, 'smoke', formats=[], cot=True); print('COT', round(time.time() - t, 1), 's', flush=True)
print('budget', A.BUDGET, 'peak GB', round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
