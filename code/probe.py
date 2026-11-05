import random, atlas, torch
tok, model = atlas.load()
def mk(k, lo, hi, far, n=60):
    r = random.Random(f'{k}-{lo}-{hi}-{far}'); out = []
    for _ in range(n):
        xs = [r.randint(lo, hi) for _ in range(k + 1)]; ans = sum(xs); ds = set()
        offs = [-40, -30, -20, -13, 13, 20, 30, 40, 7, -7] if far else [-11, -10, -9, -2, -1, 1, 2, 9, 10, 11]
        while len(ds) < 3:
            x = ans + r.choice(offs)
            if x > 0: ds.add(x)
        out.append({'q': f"What is {' + '.join(map(str, xs))}?", 'ans': str(ans), 'dist': [str(x) for x in ds], 'kind': 'number'})
    return out
for lo, hi in [(11, 49), (2, 20)]:
    for far in (0, 1):
        sets = {f'ARITH{k}': (mk(k, lo, hi, far), 'p') for k in (1, 2, 3, 4)}
        ev = atlas.evaluate(model, tok, sets)
        print(f'range {lo}-{hi} far={far}', {k: round(v['acc'], 2) for k, v in ev.items()}, flush=True)
