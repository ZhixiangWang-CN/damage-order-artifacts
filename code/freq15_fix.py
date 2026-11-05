"""Audit of the capital co-occurrence counts (reviewer comments): Italy-Rome and Bulgaria-Sofia returned 0 under the default
approximation of the infini-gram CNF count. Re-query every country with max_clause_freq=500000 -> freq15_fix.json (freq15.json is not changed)."""
import json, urllib.request, time
from scipy.stats import spearmanr
F = json.load(open('freq15.json'))['counts']; out = {}
for c, v in sorted(F.items()):
    d = {'index': 'v4_dolma-v1_7_llama', 'query_type': 'count', 'query': f"{c} AND {v['capital']}", 'max_clause_freq': 500000}
    for a in range(5):
        try: r = json.loads(urllib.request.urlopen(urllib.request.Request('https://api.infini-gram.io/', data=json.dumps(d).encode(), headers={'Content-Type': 'application/json'}), timeout=120).read()); break
        except Exception: time.sleep(3)
    out[c] = {**v, 'cooc_default': v['cooc'], 'cooc': r['count'], 'cooc_approx': r.get('approx')}; time.sleep(.3)
json.dump({'index': 'v4_dolma-v1_7_llama', 'max_clause_freq': 500000, 'counts': out}, open('freq15_fix.json', 'w'), indent=1)
ks = [c for c in out if c not in ('Italy', 'Bulgaria')]
print('Spearman old vs new (76 countries):', spearmanr([F[c]['cooc'] for c in ks], [out[c]['cooc'] for c in ks])[0])
print('zeros now:', [c for c in out if out[c]['cooc'] == 0]); print({c: out[c]['cooc'] for c in ('Italy', 'Bulgaria', 'France', 'Spain')})
