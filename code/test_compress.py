"""Unit tests for compress.py on tiny random models (CPU, no downloads). Run: python3 test_compress.py"""
import torch, torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
import compress as C

torch.manual_seed(0)

def test_matrix_level():
    W = torch.randn(64, 256); X = torch.randn(4096, 256) * torch.linspace(0.2, 3, 256); H = 2 * X.t() @ X / 4096; sq = (X ** 2).sum(0)
    for s in (0.3, 0.5, 0.8):
        Ww = C.wanda_prune(W, sq, s); assert ((Ww == 0).sum(1) == int(256 * s)).all(), 'wanda: exact per-row sparsity'
        Ws = C.sparsegpt_prune(W, H, s); frac = (Ws == 0).float().mean().item(); assert abs(frac - s) < 0.01, ('sparsegpt sparsity', frac)
        Wm = W.clone(); thr = W.abs().flatten().sort()[0][int(W.numel() * s)]; Wm[W.abs() <= thr] = 0   # naive magnitude pruning
        e = lambda A: ((X @ (A - W).t()) ** 2).mean().item()
        assert e(Ws) < e(Wm) and e(Ww) < e(Wm), ('importance-guided pruning should beat magnitude pruning on output error', s, e(Ws), e(Ww), e(Wm))
    for d, (b, g) in C.GPTQ_CFG.items():
        Wq = C.gptq_quantize(W, H, b, g); maxq = 2 ** (b - 1) - 1
        G = g if g else 256
        for j in range(0, 256, G):
            blk = Wq[:, j:j + G]; sc = blk.abs().amax(1, keepdim=True).clamp_min(1e-8)
            lv = torch.unique((blk / sc * maxq).round(decimals=3), dim=None).numel()   # values per row lie on a grid of <= 2*maxq+1 levels
        rtn = W.clone()
        for j in range(0, 256, G):
            sc = rtn[:, j:j + G].abs().amax(1, keepdim=True) / maxq; rtn[:, j:j + G] = (rtn[:, j:j + G] / sc).round().clamp(-maxq, maxq) * sc
        e = lambda A: ((X @ (A - W).t()) ** 2).mean().item()
        assert e(Wq) <= e(rtn) * 1.05, ('gptq should not be worse than round-to-nearest', b, g, e(Wq), e(rtn))
        for r in range(4):   # each row inside each group uses at most 2*maxq+1 distinct values
            for j in range(0, 256, G): assert torch.unique(Wq[r, j:j + G]).numel() <= 2 * maxq + 1, ('levels', b, g)
    print('matrix-level tests passed')

def tiny(cls, cfgcls):
    cfg = cfgcls(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return cls(cfg).eval()

def test_driver():
    for cls, cfgcls in ((Qwen2ForCausalLM, Qwen2Config), (LlamaForCausalLM, LlamaConfig)):
        for method, dose in (('wanda', 0.5), ('sparsegpt', 0.5), ('gptq', 4), ('gptq', 9)):
            m = tiny(cls, cfgcls); ids = torch.randint(0, 320, (8, 32)); before = {n: p.clone() for n, p in m.named_parameters()}
            ref = m(ids).logits
            C.compress(m, method, dose, ids)
            for n, p in m.named_parameters():
                inside = n.startswith('model.layers.') and p.dim() == 2
                if not inside: assert torch.equal(p, before[n]), ('touched outside the decoder layers', n)
                elif method != 'gptq': assert abs((p == 0).float().mean().item() - dose) < 0.02, (method, n, (p == 0).float().mean().item())
                else: assert not torch.equal(p, before[n]), ('gptq left a matrix unchanged', n)
            out = m(ids).logits; assert torch.isfinite(out).all(), 'non-finite logits'
            print(cls.__name__, method, dose, 'ok; logit change', round((out - ref).abs().mean().item(), 4))

if __name__ == '__main__':
    test_matrix_level(); test_driver(); print('ALL PASSED')
