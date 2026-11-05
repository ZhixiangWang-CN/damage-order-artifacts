"""Importance-guided compression for PROTOCOL_ADDENDUM_9: Wanda, SparseGPT and GPTQ, applied layer by layer to every linear layer
inside the decoder layers (embeddings and the output head are untouched).

Follows the reference implementations (Sun et al. 2024; Frantar & Alistarh 2023; Frantar et al. 2023): calibration inputs are passed
through the decoder layers in order, and each layer is compressed from the outputs of the already-compressed layers before it. All
linear layers of one decoder layer are compressed from the same inputs (one forward pass per layer for the statistics).
GPTQ here is symmetric round-to-nearest with one scale per group (no zero point), act-order off, as written in the protocol.
"""
import gzip, json, random
import torch, torch.nn as nn

BLOCK = 128; DAMP = 0.01
class _Stop(Exception): pass
GPTQ_CFG = {1: (4, 128), 2: (4, 0), 3: (3, 32), 4: (3, 128), 5: (3, 0), 6: (2, 32), 7: (2, 64), 8: (2, 128), 9: (2, 0)}   # dose index -> (bits, group); 0 = per channel


# ---------------------------------------------------------------- calibration text
def c4_calibration(tok, path, n=128, seqlen=512, seed=0):
    """n windows of seqlen tokens from random C4 documents longer than seqlen (as in the GPTQ / SparseGPT reference code)."""
    with gzip.open(path, 'rt') as fh: docs = [json.loads(l)['text'] for l in fh]
    rng = random.Random(seed); out = []
    while len(out) < n:
        ids = tok(docs[rng.randrange(len(docs))], add_special_tokens=False).input_ids
        if len(ids) <= seqlen: continue
        st = rng.randrange(len(ids) - seqlen); out.append(ids[st:st + seqlen])
    return torch.tensor(out)


# ---------------------------------------------------------------- per-matrix algorithms (all in float32 on the matrix's device)
def _hinv_upper(H):
    H = H.clone(); dead = torch.diag(H) == 0; H[dead, dead] = 1
    idx = torch.arange(H.shape[0], device=H.device); H[idx, idx] += DAMP * torch.mean(torch.diag(H))
    H = torch.linalg.cholesky(H); H = torch.cholesky_inverse(H); return torch.linalg.cholesky(H, upper=True), dead

def wanda_prune(W, sq_norm, s):
    """prune the fraction s of each output row with the smallest |W| * ||x||."""
    metric = W.abs() * torch.sqrt(sq_norm).reshape(1, -1); k = int(W.shape[1] * s)
    if k == 0: return W
    idx = torch.sort(metric, dim=-1, stable=True)[1][:, :k]; W = W.clone(); W.scatter_(1, idx, 0); return W

def sparsegpt_prune(W, H, s):
    W = W.clone(); Hinv, dead = _hinv_upper(H); W[:, dead] = 0; cols = W.shape[1]
    for i1 in range(0, cols, BLOCK):
        i2 = min(i1 + BLOCK, cols); n = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape(1, -1)) ** 2
        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * s)] if s > 0 else -1.0
        mask1 = tmp <= thresh
        for i in range(n):
            w = W1[:, i]; d = Hinv1[i, i]; q = w.clone(); q[mask1[:, i]] = 0
            Q1[:, i] = q; err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)); Err1[:, i] = err
        W[:, i1:i2] = Q1; W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W

def _scale(x, maxq):
    return (x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / maxq)

def gptq_quantize(W, H, bits, group):
    W = W.clone(); maxq = 2 ** (bits - 1) - 1; Hinv, dead = _hinv_upper(H); W[:, dead] = 0; cols = W.shape[1]
    sc = _scale(W, maxq) if group == 0 else None
    for i1 in range(0, cols, BLOCK):
        i2 = min(i1 + BLOCK, cols); n = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(n):
            if group and (i1 + i) % group == 0: sc = _scale(W[:, (i1 + i):(i1 + i + group)], maxq)   # as in the reference code
            w = W1[:, i]; d = Hinv1[i, i]; q = (torch.clamp(torch.round(w.unsqueeze(1) / sc), -maxq, maxq) * sc).flatten()
            Q1[:, i] = q; err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)); Err1[:, i] = err
        W[:, i1:i2] = Q1; W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W


# ---------------------------------------------------------------- sequential driver
def _linears(layer): return [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]
def _dev(layer): return next(layer.parameters()).device

def _move(v, d):
    if torch.is_tensor(v): return v.to(d)
    if isinstance(v, tuple): return tuple(_move(x, d) for x in v)
    return v

@torch.no_grad()
def compress(model, method, dose, calib_ids):
    """apply one compression in place. dose: sparsity (wanda, sparsegpt) or GPTQ_CFG index (gptq)."""
    layers = model.model.layers; first = _dev(layers[0]); store = {'x': [], 'kw': None}
    def catch(mod, args, kwargs):
        store['x'].append(args[0] if args else kwargs['hidden_states']); store['kw'] = {k: v for k, v in kwargs.items() if k != 'hidden_states'}; raise _Stop
    h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    for i in range(calib_ids.shape[0]):
        try: model(calib_ids[i:i + 1].to(first), use_cache=False)
        except _Stop: pass
    h.remove(); xs = store['x']; kw = store['kw']; kw['use_cache'] = False; kw.pop('past_key_value', None); kw.pop('past_key_values', None)
    for layer in layers:
        d = _dev(layer); kwd = {k: _move(v, d) for k, v in kw.items()}; lins = _linears(layer)
        stats = {n: None for n, _ in lins}; hooks = []
        def mk(n):
            def f(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                v = (x ** 2).sum(0) if method == 'wanda' else x.t().matmul(x)
                stats[n] = v if stats[n] is None else stats[n] + v
            return f
        for n, m in lins: hooks.append(m.register_forward_hook(mk(n)))
        for x in xs: layer(x.to(d), **kwd)
        for hk in hooks: hk.remove()
        for n, m in lins:
            W = m.weight.data.float()
            if method == 'wanda': Wn = wanda_prune(W, stats[n], dose)
            elif method == 'sparsegpt': Wn = sparsegpt_prune(W, stats[n] * (2.0 / len(xs)), dose)
            elif method == 'gptq': b, g = GPTQ_CFG[int(dose)]; Wn = gptq_quantize(W, stats[n] * (2.0 / len(xs)), b, g)
            else: raise ValueError(method)
            m.weight.data.copy_(Wn.to(m.weight.dtype)); stats[n] = None
        new = []
        for x in xs:
            o = layer(x.to(d), **kwd); new.append(o[0] if isinstance(o, tuple) else o)
        xs = new
        torch.cuda.empty_cache()
