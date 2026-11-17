"""Compare seed-level LM1B-small metrics under the frozen quality protocol.

This compares already verified metrics only. It is not a scorer, does not
validate generation or model identities, and cannot alone establish a gate.
"""
import math
import random
import statistics

SEEDS = (42, 0, 1, 2, 3)
PPL_MARGIN = 5.46
ENTROPY_MARGIN = 0.0046


def checked_values(records):
    if not isinstance(records, list) or len(records) != len(SEEDS):
        raise ValueError('exactly five seed records required')
    found = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {'seed', 'count', 'ppl', 'entropy'}:
            raise ValueError('invalid score record schema')
        seed = record['seed']
        if type(seed) is not int or seed not in SEEDS or seed in found or type(record['count']) is not int or record['count'] != 256:
            raise ValueError('invalid, duplicate or incomplete seed')
        values = (record['ppl'], record['entropy'])
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError('nonfinite or nonnumeric metric')
        if values[0] <= 0 or values[1] < 0:
            raise ValueError('invalid metric range')
        found[seed] = values
    if set(found) != set(SEEDS):
        raise ValueError('missing official seed')
    return [found[seed] for seed in SEEDS]


def compare(reference, candidate):
    left, right = checked_values(reference), checked_values(candidate)
    p_left, e_left = zip(*left, strict=True)
    p_right, e_right = zip(*right, strict=True)
    point = {'ppl': statistics.mean(p_right) - statistics.mean(p_left),
             'entropy': statistics.mean(e_right) - statistics.mean(e_left)}
    rng = random.Random(42)
    # Draw the same seed indices for both metrics within a runtime, preserving
    # per-seed association while keeping the two runtimes independent.
    draws = {'ppl': [], 'entropy': []}
    for _ in range(10000):
        a = [rng.randrange(5) for _ in range(5)]
        b = [rng.randrange(5) for _ in range(5)]
        draws['ppl'].append(statistics.mean(p_right[i] for i in b) - statistics.mean(p_left[i] for i in a))
        draws['entropy'].append(statistics.mean(e_right[i] for i in b) - statistics.mean(e_left[i] for i in a))
    intervals = {key: [sorted(values)[249], sorted(values)[9749]] for key, values in draws.items()}
    checks = {'ppl_mean': point['ppl'] <= PPL_MARGIN,
              'entropy_mean': abs(point['entropy']) <= ENTROPY_MARGIN,
              'ppl_interval': intervals['ppl'][1] <= PPL_MARGIN,
              'entropy_interval': all(abs(x) <= ENTROPY_MARGIN for x in intervals['entropy'])}
    return {'scope': 'metric comparison only; generation, scorer and model identities require separate validation',
            'seed_order': list(SEEDS), 'difference': point, 'interval_95': intervals,
            'margins': {'ppl': PPL_MARGIN, 'entropy': ENTROPY_MARGIN},
            'bootstrap': {'method': 'independent seed resampling', 'seed': 42, 'resamples': 10000},
            'checks': checks, 'metrics_passed': all(checks.values())}
