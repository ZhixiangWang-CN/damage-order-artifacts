"""Unit tests for the judges of atlas_real.py (run locally; no model)."""
import json, atlas_real as R
it = {'function': {'name': 'find_recipe', 'parameters': {'properties': {'maxCalories': {'type': 'integer'}, 'recipeName': {'type': 'string'}}, 'required': ['recipeName']}},
      'gt': {'find_recipe': {'maxCalories': [500], 'recipeName': ['pasta carbonara']}}}
cases = [('{"name": "find_recipe", "arguments": {"recipeName": "Pasta Carbonara", "maxCalories": 500}}', (1, 1, 1)),
         ('```json\n{"name": "find_recipe", "arguments": {"recipeName": "pasta carbonara", "maxCalories": 500}}\n```', (1, 1, 1)),
         ('{"name": "find_recipe", "arguments": {"recipeName": "pasta carbonara"}}', (0, 1, 0)),        # maxCalories required by the answer, absent
         ('Sure! {"name": "find_recipe", "arguments": {"recipeName": "pasta carbonara", "maxCalories": 500}}', (0, 0, 1)),
         ('find_recipe(recipeName="pasta carbonara", maxCalories=500)', (0, 0, 1)),
         ('{"name": "find_recipe", "arguments": {"recipeName": "pizza", "maxCalories": 500}}', (0, 1, 0)),
         ('{"name": "find_recipe", "arguments": {"dish": "pasta carbonara", "maxCalories": 500}}', (0, 0, 1)),
         ('{"name": "find_recipe", "arguments": {"recipeName": "pasta carbonara", "maxCalories": 500.0}}', (1, 1, 1))]
for out, exp in cases:
    got = R.judge_fcall(it, out); assert got == exp, (out, got, exp)
it2 = {'function': {'name': 'f', 'parameters': {'properties': {'a': {}, 'b': {}}}}, 'gt': {'f': {'a': [1], 'b': ['', 'x']}}}
assert R.judge_fcall(it2, '{"name": "f", "arguments": {"a": 1}}') == (1, 1, 1)   # optional argument omitted
it3 = {'function': {'name': 'g', 'parameters': {'properties': {'p': {}}}}, 'gt': {'g': {'p': [[3.0, 2.0]]}}}
assert R.judge_fcall(it3, '{"name": "g", "arguments": {"p": [3, 2]}}') == (1, 1, 1)   # 3 and 3.0 are the same number
he = {'src': 'humaneval', 'prompt': 'def add(a, b):\n    """add"""\n', 'test': 'def check(f):\n    assert f(1, 2) == 3\ncheck(add)\n'}
assert R.judge_code(he, '```python\ndef add(a, b):\n    return a + b\n```') == 1
assert R.judge_code(he, '```python\n    return a + b\n```') == 1                    # body only
assert R.judge_code(he, '```python\ndef add(a, b):\n    return a - b\n```') == 0
assert R.judge_code(he, 'def add(a, b):\n    while True: pass') == 0                  # timeout
assert R.judge_code(he, '```python\nimport os\nos.system("echo hi")\ndef add(a,b): return a+b\n```') == 0   # guarded call fails
mb = {'src': 'mbpp', 'prompt': 'x', 'tests': ['assert sq(3) == 9'], 'test': '\nassert sq(3) == 9\n'}
assert R.judge_code(mb, 'Here:\n```python\ndef sq(x):\n    return x * x\n```\nDone.') == 1
assert R.judge_all('GSM_COT', [{'ans': '42'}], ['so the answer is 42.'])['correct'] == [1]
print('all judge tests passed')
