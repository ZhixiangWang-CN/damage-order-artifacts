"""Intact-model confidence per measure (CONFIDENCE_PLAN.md). python3 conf_server.py <name>   with ATLAS_MODEL set."""
import sys, json, torch, numpy as np
import atlas, atlas_bench as B, atlas_fmt as F
from atlas import load
name = sys.argv[1]; tok, model = load(); eos = tok.eos_token_id
@torch.no_grad()
def greedy_conf(prompts, mx):
    outs, nll, mar = [], [], []
    for i in range(0, len(prompts), 40):
        enc = tok(prompts[i:i + 40], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
        seq = g.sequences[:, enc['input_ids'].shape[1]:]; sc = torch.stack(g.scores, 1).float()          # [b, steps, vocab]
        lp = sc.log_softmax(-1); top2 = sc.topk(2, -1).values; chosen = lp.gather(-1, seq.unsqueeze(-1)).squeeze(-1); gap = top2[..., 0] - top2[..., 1]
        for j in range(seq.shape[0]):
            e = (seq[j] == eos).nonzero(); L = int(e[0]) + 1 if len(e) else seq.shape[1]
            nll.append(float(-chosen[j, :L].sum())); mar.append(float(gap[j, :L].min()))
        outs += tok.batch_decode(seq, skip_special_tokens=True)
    return outs, np.array(nll), np.array(mar)
res = {}
sets = {t: F.items_for(t, 200, 'eval') for t in F.CONTENT + ['COPYLONG']}
for t in F.CONTENT:
    for f in F.FMTS:
        its = sets[t]; o, nll, mar = greedy_conf([F.chat(tok, it['q'] + '\n' + F.INSTR[f].format(n=F.NOUN[t])) for it in its], 24)
        ok = np.array([F.format_ok(f, x) and F.content_ok(t, x, it['ans']) for x, it in zip(o, its)], bool)
        res[f'{t}_{f}'] = {'P1': float(nll[ok].mean()), 'P2': float(mar[ok].mean()), 'n_ok': int(ok.sum())}
its = sets['COPYLONG']; o, nll, mar = greedy_conf([F.chat(tok, it['q']) for it in its], 40); ok = np.array([F.content_ok('COPYLONG', x, it['ans']) for x, it in zip(o, its)], bool)
res['COPYLONG'] = {'P1': float(nll[ok].mean()), 'P2': float(mar[ok].mean()), 'n_ok': int(ok.sum())}
end = [eos]
for t in B.OLD:
    its = B.get_items(t, 200, 'eval'); ps, gold = B.prompts_for(tok, t, its, 'GEN', 'eval')
    o, nll, mar = greedy_conf(ps, 12); ok = np.array([B.judge(t, 'GEN', it, x, g) for it, x, g in zip(its, o, gold)], bool)
    res[f'{t}_GEN'] = {'P1': float(nll[ok].mean()), 'P2': float(mar[ok].mean()), 'n_ok': int(ok.sum())}
    p1, p2 = [], []
    with torch.no_grad():
        for i in range(0, len(ps), 16):
            seqs, spans = [], []
            for p, it in zip(ps[i:i + 16], its[i:i + 16]):
                pid = tok.encode(p, add_special_tokens=False)
                for c in it['cands']:
                    cid = tok.encode(c, add_special_tokens=False) + end; seqs.append(pid + cid); spans.append(len(cid))
            m = max(map(len, seqs)); ids = torch.full((len(seqs), m), tok.pad_token_id); att = torch.zeros((len(seqs), m), dtype=torch.long)
            for j, sq in enumerate(seqs): ids[j, m - len(sq):] = torch.tensor(sq); att[j, m - len(sq):] = 1
            W = max(spans) + 1; lp = model.lm_head(model.model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).last_hidden_state[:, -W:, :]).float().log_softmax(-1)
            tot = torch.tensor([sum(lp[j, W - n - 1 + q, seqs[j][len(seqs[j]) - n + q]].item() for q in range(n)) for j, n in enumerate(spans)]); k = len(its[0]['cands'])
            for a in range(0, len(tot), k):
                v = tot[a:a + k]
                if int(v.argmax()) == 0: p1.append(float(-(v.log_softmax(0)[0]))); p2.append(float(v[0] - v[1:].max()))
    res[f'{t}_CHOICE'] = {'P1': float(np.mean(p1)), 'P2': float(np.mean(p2)), 'n_ok': len(p1)}
json.dump(res, open(f'conf_{name}.json', 'w'), indent=1)
print(name, {k: (round(v['P1'], 4), round(v['P2'], 2), v['n_ok']) for k, v in res.items()})
