"""Diagnostic for the addendum-9 smoke check: importance-guided compression vs naive baselines (magnitude pruning, round-to-nearest)."""
import sys, json, torch
import atlas_comp as AC, compress as C
from atlas_comp import load_model, Dmg, ppl
key = sys.argv[1]; wiki = 'data/wikitext2-test.parquet'; AC.C4 = 'data/c4-validation.00000.json.gz'
tok, model, end = load_model(key); dm = Dmg(model); out = {'intact': ppl(model, tok, wiki)}; print(key, 'intact', round(out['intact'], 3), flush=True)
def mag(s):
    with torch.no_grad():
        for n, p in dm.params:
            k = int(p.shape[1] * s); idx = torch.sort(p.abs(), dim=-1, stable=True)[1][:, :k]; p.scatter_(1, idx, 0)
def rtn(bits, g):
    q = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for n, p in dm.params:
            w = p.float(); o, i = w.shape; G = g if g else i; w = w.view(o, i // G, G); sc = w.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
            p.copy_(((w / sc).round().clamp(-q, q) * sc).view(o, i).to(p.dtype))
for name, f in [('magnitude_0.5', lambda: mag(0.5)), ('wanda_0.5', lambda: dm.apply({'damage': 'wanda', 'dose': 0.5, 'seed': 0})),
                ('sparsegpt_0.5', lambda: dm.apply({'damage': 'sparsegpt', 'dose': 0.5, 'seed': 0})),
                ('rtn_4b_g128', lambda: rtn(4, 128)), ('gptq_4b_g128', lambda: dm.apply({'damage': 'gptq', 'dose': 1, 'seed': 0})),
                ('rtn_3b_g128', lambda: rtn(3, 128)), ('gptq_3b_g128', lambda: dm.apply({'damage': 'gptq', 'dose': 4, 'seed': 0}))]:
    dm.restore(); f(); out[name] = ppl(model, tok, wiki); print(key, name, round(out[name], 3), 'rel', round(out[name] / out['intact'] - 1, 3), flush=True)
dm.restore(); json.dump(out, open(f'runs/smoke/diag_{key}.json', 'w'), indent=1)
