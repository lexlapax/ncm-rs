"""Summarize pinned LM1B development seeds before official quality evaluation."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

from tinystories_artifact import ROOT, digest_file, json_file, verify_input

CAPTURES = {
    11: '123ce052e32ebac1981321a72d728c7b16c876be5c8a118dd4668b0dc917760a',
    12: '72f802a08efcae3512b6e9fdc9a70c2044c9092b71397ad0a0e265dd440e88c3',
    13: '4a3683e8c99c459d5a02f468c91c0121067b12ce85efa96a86453958708ffc36',
    14: 'c0c438eaf647d7e9ca9ee9cf70b2d04b2ee9fae8faafeadfc3d7a1cbc04caa85',
    15: '3a07c39863262f745ad61d4cbb170ec081f93c7988d960afbc352eadd202cb26',
}
SCORES = {
    11: 'fcbc18439c7207e7ed650ef3c95c0da360fb331dc2b28a61bcdb4eaa78605054',
    12: 'a2783868ddbb1a777cd1ccf35335a071b15b2a8e4e58983cd9e58ec9d5681de5',
    13: '55d30dcdc23dba3a6f07a0317e834644a4b8dac1efdd8e19cd8a8006044f5357',
    14: 'b44b184dcae8ce8ff430c45a32c0d0f51f62cfe9d39519ccdbcfca998b038684',
    15: 'd23be7576cc1ea9ea42f20d133276082ccd9bfaec102385a71a8d2148ffcca40',
}


def summary(values, rounding):
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError('five finite seed values required')
    rng = random.Random(42)
    means = sorted(statistics.mean(rng.choices(values, k=5)) for _ in range(10000))
    sd = statistics.stdev(values)
    return {'mean': statistics.mean(values), 'sample_sd': sd,
            'seed_bootstrap_95': [means[249], means[9749]],
            'twice_sd_rounded_up': math.ceil(2 * sd * rounding) / rounding}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    watched = [Path(__file__).resolve()]
    values = []
    for seed in CAPTURES:
        capture = ROOT / f'reports/phase4-lm1b-small-dev-seed-{seed}-v1.json'
        score = ROOT / f'reports/phase4-lm1b-small-dev-score-{seed}-v1.json'
        verify_input(capture, capture.stat().st_size, CAPTURES[seed])
        verify_input(score, score.stat().st_size, SCORES[seed])
        item, metric = json.loads(capture.read_bytes()), json.loads(score.read_bytes())
        if (item['size'], item['seed'], item['count'], item['batch_size']) != ('small', seed, 256, 8):
            raise ValueError('development capture metadata differs')
        if (metric['count'] != 256 or metric['upstream_exact'] is not True or
                metric['sample_sha256'] != {'development': CAPTURES[seed]} or len(metric['entropy']) != 256):
            raise ValueError('development score metadata differs')
        values.append({'seed': seed, 'ppl': metric['ppl'], 'entropy': statistics.mean(metric['entropy'])})
        watched.extend((capture, score))
    before = {str(path): digest_file(path) for path in watched}
    result = {'scope': 'Python-only development baseline; not held-out or Rust quality',
              'rows_per_seed': 256, 'seeds': list(CAPTURES), 'values': values,
              'ppl': summary([v['ppl'] for v in values], 100),
              'entropy': summary([v['entropy'] for v in values], 10000),
              'bootstrap': {'unit': 'seed mean', 'resamples': 10000, 'seed': 42,
                            'method': 'percentile, five seeds resampled with replacement'},
              'capture_sha256': CAPTURES, 'score_sha256': SCORES}
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed while summarizing')
    for seed in CAPTURES:
        capture = ROOT / f'reports/phase4-lm1b-small-dev-seed-{seed}-v1.json'
        score = ROOT / f'reports/phase4-lm1b-small-dev-score-{seed}-v1.json'
        verify_input(capture, capture.stat().st_size, CAPTURES[seed])
        verify_input(score, score.stat().st_size, SCORES[seed])
    json_file(args.output, result)
    print(json.dumps({'ppl': result['ppl'], 'entropy': result['entropy'],
                      'sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
