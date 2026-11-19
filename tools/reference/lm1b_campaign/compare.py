"""Development preflight for expanding independent LM1B seed resampling."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from lm1b_quality_compare import ENTROPY_MARGIN, PPL_MARGIN, SEEDS, compare

RUNTIMES = ('python', 'rust')
ARCHIVES = {
    'original': ('phase4-lm1b-small-heldout-cuda-v1',
                 '2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002'),
    'efficient': ('phase4-lm1b-efficient-attention-v1.xYWlWf/retest',
                  'dc3ab67eb9031b10774202346fec1b0cf90a47d7cbc2b7453b9e8a9c2e66f888'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_archive_report(name):
    directory, expected = ARCHIVES[name]
    root = Path('reports') / directory
    source = root / 'summary.json'
    if digest(source) != expected:
        raise ValueError(f'{name} archive identity differs')
    return root, json.loads(source.read_bytes())


def checked_values(records, seeds):
    if (len(seeds) < 2 or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('invalid ordered seed list')
    records_by_seed = {}
    for record in records:
        if set(record) != {'seed', 'count', 'ppl', 'entropy'}:
            raise ValueError('invalid metric schema')
        seed, count = record['seed'], record['count']
        if type(seed) is not int or type(count) is not int or count != 256 or seed in records_by_seed:
            raise ValueError('invalid or duplicate seed record')
        values = (record['ppl'], record['entropy'])
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or values[0] <= 0 or values[1] < 0):
            raise ValueError('invalid metric')
        records_by_seed[seed] = values
    if set(records_by_seed) != set(seeds):
        raise ValueError('seed records do not match manifest')
    return [records_by_seed[seed] for seed in seeds]


def compare_cohorts(records, seed_order):
    if set(records) != set(RUNTIMES) or set(seed_order) != set(RUNTIMES):
        raise ValueError('exactly two fixed runtime cohorts required')
    python = checked_values(records['python'], seed_order['python'])
    rust = checked_values(records['rust'], seed_order['rust'])
    if len(python) != len(rust):
        raise ValueError('cohorts must have the same sample size')

    left, right = list(zip(*python, strict=True)), list(zip(*rust, strict=True))
    keys = ('ppl', 'entropy')
    point = {key: statistics.mean(right[i]) - statistics.mean(left[i])
             for i, key in enumerate(keys)}
    rng = random.Random(42)
    draws = {key: [] for key in keys}
    n = len(python)
    for _ in range(10_000):
        a = [rng.randrange(n) for _ in range(n)]
        b = [rng.randrange(n) for _ in range(n)]
        for i, key in enumerate(keys):
            draws[key].append(statistics.mean(right[i][j] for j in b) -
                              statistics.mean(left[i][j] for j in a))
    intervals = {key: [sorted(values)[249], sorted(values)[9749]] for key, values in draws.items()}
    checks = {'ppl_mean': point['ppl'] <= PPL_MARGIN,
              'entropy_mean': abs(point['entropy']) <= ENTROPY_MARGIN,
              'ppl_interval': intervals['ppl'][1] <= PPL_MARGIN,
              'entropy_interval': all(abs(x) <= ENTROPY_MARGIN for x in intervals['entropy'])}
    return {'difference': point, 'interval_95': intervals,
            'margins': {'ppl': PPL_MARGIN, 'entropy': ENTROPY_MARGIN},
            'bootstrap': {'method': 'independent seed resampling', 'seed': 42, 'resamples': 10_000},
            'checks': checks, 'metrics_passed': all(checks.values())}


def run_controls(output):
    cases = {}
    for label in ARCHIVES:
        _, archive = read_archive_report(label)
        records = archive['records']
        frozen = archive['comparison']
        assert frozen == compare(records['python'], records['rust'])
        result = compare_cohorts(records, dict.fromkeys(RUNTIMES, SEEDS))
        assert all(result[key] == frozen[key] for key in result)
        cases[label] = result

    # These labels identify fabricated metric records. No generator is called.
    seed_order = {'python': list(range(100, 180)), 'rust': list(range(300, 380))}
    records = {key: [{'seed': seed, 'count': 256, 'ppl': 100.0, 'entropy': 4.0}
                     for seed in seed_order[key]] for key in RUNTIMES}
    varying = {key: [dict(record, entropy=4.0 + ((i % 2) * 2 - 1) * 0.05)
                     for i, record in enumerate(rows)] for key, rows in records.items()}
    cases['synthetic_equal_80'] = compare_cohorts(records, seed_order)
    cases['synthetic_varying_80'] = compare_cohorts(varying, seed_order)
    assert cases['synthetic_equal_80']['metrics_passed']
    assert cases['synthetic_varying_80']['checks']['entropy_mean']
    assert not cases['synthetic_varying_80']['checks']['entropy_interval']
    assert cases['synthetic_varying_80'] == compare_cohorts(
        {key: list(reversed(rows)) for key, rows in varying.items()}, seed_order)
    damaged = {
        'missing': records['rust'][:-1],
        'duplicate': records['rust'][:-1] + records['rust'][:1],
        'wrong_count': [dict(records['rust'][0], count=True)] + records['rust'][1:],
        'nonfinite': [dict(records['rust'][0], entropy=math.nan)] + records['rust'][1:],
        'out_of_manifest': [dict(records['rust'][0], seed=600)] + records['rust'][1:],
    }
    rejects = Counter()
    for name, bad in damaged.items():
        try:
            compare_cohorts(dict(records, rust=bad), seed_order)
        except ValueError:
            rejects[name] += 1
        else:
            raise AssertionError(f'accepted {name}')
    with output.open('x') as stream:
        json.dump({'scope': 'development comparator controls, no new captures', 'cases': cases,
                   'rejected': dict(rejects)}, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    run_controls(Path(sys.argv[1]))
