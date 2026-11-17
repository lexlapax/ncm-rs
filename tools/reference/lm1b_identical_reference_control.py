"""Descriptive identical-reference control, not a candidate qualification."""
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root = Path('reports/phase4-lm1b-small-heldout-cuda-v1')
summary_bytes = (root / 'summary.json').read_bytes()
assert hashlib.sha256(summary_bytes).hexdigest() == '2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002'
summary = json.loads(summary_bytes)
spec = importlib.util.spec_from_file_location('quality_compare', 'tools/reference/lm1b_quality_compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
rows = []
for seed in compare.SEEDS:
    path = root / f'python-seed-{seed}-score.json'
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == summary['identities'][f'python:{seed}']['score_sha256']
    score = json.loads(data)
    assert score['seed'] == seed and score['count'] == 256 and score['upstream_exact']
    rows.append({'seed': seed, 'count': score['count'], 'ppl': score['ppl'], 'entropy': statistics.mean(score['entropy'])})
result = compare.compare(rows, rows)
print(json.dumps(result, sort_keys=True, indent=2))
