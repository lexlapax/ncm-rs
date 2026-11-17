"""Descriptive exhaustive splits of observed Python seeds, not new held-out data."""
import hashlib
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, 'tools/reference')
from lm1b_dev_protocol import CAPTURES, SCORES
from lm1b_quality_compare import SEEDS, compare

root = Path('reports')
summary_path = root / 'phase4-lm1b-small-heldout-cuda-v1/summary.json'
watched = [summary_path]
summary = summary_path.read_bytes()
assert hashlib.sha256(summary).hexdigest() == '2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002'
held = json.loads(summary)['records']['python']
assert [x['seed'] for x in held] == list(SEEDS)
assert all(x['count'] == 256 for x in held)
data = [(x['ppl'], x['entropy']) for x in held]
for seed in range(11, 16):
    cap_path = root / f'phase4-lm1b-small-dev-seed-{seed}-v1.json'
    score_path = root / f'phase4-lm1b-small-dev-score-{seed}-v1.json'
    watched.extend([cap_path, score_path])
    cap = cap_path.read_bytes()
    score = score_path.read_bytes()
    assert hashlib.sha256(cap).hexdigest() == CAPTURES[seed]
    assert hashlib.sha256(score).hexdigest() == SCORES[seed]
    c, m = json.loads(cap), json.loads(score)
    assert (c['size'], c['seed'], c['count'], c['batch_size']) == ('small', seed, 256, 8)
    assert m['count'] == 256 and m['upstream_exact'] is True and len(m['entropy']) == 256
    assert m['sample_sha256'] == {'development': CAPTURES[seed]}
    data.append((m['ppl'], statistics.mean(m['entropy'])))
before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
rng = random.Random(42)
a, b = [], []
for _ in range(10000):
    a.append([rng.randrange(5) for _ in range(5)])
    b.append([rng.randrange(5) for _ in range(5)])
a, b = np.array(a), np.array(b)
values = np.array(data)
failure = {'entropy_interval': 0, 'entropy_mean': 0, 'ppl_interval': 0, 'ppl_mean': 0, 'any': 0}
checked = 0
for index, indices in enumerate(itertools.combinations(range(10), 5)):
    other = [i for i in range(10) if i not in indices]
    left, right = values[list(indices)], values[other]
    differences = right.mean(axis=0) - left.mean(axis=0)
    draws = (right[b].mean(axis=1) - left[a].mean(axis=1))
    draws.sort(axis=0)
    bad = {'entropy_interval': abs(draws[249, 1]) > .0046 or abs(draws[9749, 1]) > .0046,
           'entropy_mean': abs(differences[1]) > .0046,
           'ppl_interval': draws[9749, 0] > 5.46,
           'ppl_mean': differences[0] > 5.46}
    for key, value in bad.items(): failure[key] += int(value)
    failure['any'] += int(any(bad.values()))
    if index % 31 == 0:
        records = lambda group: [dict(seed=seed, count=256, ppl=float(v[0]), entropy=float(v[1])) for seed, v in zip(SEEDS, group, strict=True)]
        exact = compare(records(left), records(right))
        for i, key in enumerate(('ppl', 'entropy')):
            assert np.allclose(draws[[249, 9749], i], exact['interval_95'][key], rtol=0, atol=1e-12)
        assert all(bad[k] == (not exact['checks'][k]) for k in bad)
        checked += 1
assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
print(json.dumps({'scope': 'overlapping partitions of ten observed Python seeds; descriptive only', 'splits': 252, 'comparator_crosschecks': checked, 'failures': failure}, sort_keys=True))
