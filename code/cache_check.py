"""Does cached generation stay correct after layers are removed? Compare use_cache=True vs False."""
import sys, torch, atlas, atlas_bench as B, atlas_ext as E
tok, model, dm = E.setup()
its = B.get_items('ARITH3', 24, 'chk') + B.get_items('FACT', 24, 'chk')
ps = [tok.apply_chat_template([{'role': 'user', 'content': it['q'] + '\nAnswer with just the answer.'}], tokenize=False, add_generation_prompt=True) for it in its]
def gen(uc):
    outs = []
    for i in range(0, len(ps), 24):
        enc = tok(ps[i:i + 24], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        o = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=uc)
        outs += tok.batch_decode(o[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    return outs
for c in [{'damage': 'none', 'dose': 0}, {'damage': 'ld_late', 'dose': 3}, {'damage': 'ld_rand', 'dose': 4, 'seed': 1}, {'damage': 'ld_rand', 'dose': 8, 'seed': 2}, {'damage': 'ld_mid', 'dose': 4}]:
    dm.apply(c) if c['damage'] != 'none' else dm.restore()
    try:
        a = gen(True); b = gen(False); same = sum(x == y for x, y in zip(a, b))
        print(atlas.MODEL.split('/')[-1], c, f'identical {same}/{len(a)}', '| example cached:', repr(a[0]), 'uncached:', repr(b[0]), flush=True)
    except Exception as e: print(atlas.MODEL.split('/')[-1], c, 'ERROR', type(e).__name__, str(e)[:120], flush=True)
