"""Resumable, rate-limited version of `analyze15.py freq` (same queries, same output format).
Written after the first fetch hit HTTP 403 at the 38th country; analyze15.py itself is unchanged."""
import json, os, time, urllib.request, urllib.error
from analyze15 import fact_countries, INDEX
PART = 'freq15_partial.json'
def ig(query):
    req = urllib.request.Request('https://api.infini-gram.io/', data=json.dumps({'index': INDEX, 'query_type': 'count', 'query': query}).encode(),
                                 headers={'Content-Type': 'application/json'})
    for a in range(8):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=90).read())
            if 'count' in r: return r
            err = r
        except Exception as e: err = e
        time.sleep(min(20 * (a + 1), 120))
    raise RuntimeError(f'{query}: {err}')
cs, ans = fact_countries(); pairs = sorted(set(zip(cs, ans)))
out = json.load(open(PART)) if os.path.exists(PART) else {}
for c, a in pairs:
    if c in out: continue
    co = ig(f'{c} AND {a}'); time.sleep(2); cap = ig(a); time.sleep(2); ph = ig(f'capital of {c} is {a}'); time.sleep(2)
    out[c] = {'capital': a, 'cooc': co['count'], 'cooc_approx': co.get('approx'), 'cap': cap['count'], 'cap_ntok': len(cap['tokens']), 'phrase': ph['count']}
    json.dump(out, open(PART, 'w'), indent=1); print(len(out), c, a, out[c]['cooc'], out[c]['cap'], out[c]['phrase'], flush=True)
assert len(out) == len(pairs) == 78
json.dump({'index': INDEX, 'counts': out}, open('freq15.json', 'w'), indent=1); print('done')
