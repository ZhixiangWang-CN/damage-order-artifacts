"""Exploratory: instruction push on the addendum-6 format battery.  python3 push6.py <model_key q7|mis> <out.json.gz>
Same items as atlas_fmt main (items_for(task, 200, 'eval')); canonical targets per format; push = log p(target | q + format instruction) - log p(target | q).
"""
import sys, gzip, json
import atlas_full as A
from atlas_fmt import INSTR, NOUN, CONTENT, items_for, chat
from push import score
TGT = {'F0': '{a}', 'F1': 'ANSWER: {a}', 'F2': '{{"answer": "{a}"}}', 'F3': '[[{a}]]', 'F4': '{a} DONE', 'F5': '{a}, {a}'}

def main(key, out_f):
    tok, model, end = A.load_model(key); res = {}
    for t in CONTENT:
        its = items_for(t, 200, 'eval')
        for f in TGT:
            cont = [tok.encode(TGT[f].format(a=it['ans']), add_special_tokens=False) + [end] for it in its]
            pw = [tok.encode(chat(tok, it['q'] + '\n' + INSTR[f].format(n=NOUN[t])), add_special_tokens=False) for it in its]
            po = [tok.encode(chat(tok, it['q']), add_special_tokens=False) for it in its]
            w = score(model, tok, list(zip(pw, cont))); o = score(model, tok, list(zip(po, cont)))
            res[f'{t}_{f}'] = {'with': [round(v, 4) for v in w], 'without': [round(v, 4) for v in o]}
            print(key, t, f, round(sum(a - b for a, b in zip(w, o)) / len(w), 2), flush=True)
    with gzip.open(out_f, 'wt') as fh: json.dump({'model': key, 'res': res}, fh)

if __name__ == '__main__': main(*sys.argv[1:])
