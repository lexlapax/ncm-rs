"""Summarize checked LM1B development scores; never a held-out quality gate."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

from lm1b_dev_protocol import CAPTURES, SCORES
from tinystories_artifact import ROOT, digest_file

RUST_CAPTURES = {
    11: 'fa2bf52276c1dc228a69d46f8af61ae38206f7bad05fa36a2b0571f67282b7f3',
    12: 'c7f16b03ab1dbecb060681073b48e19baeee833b13404f8f165dfbbdd5811c44',
    13: '3e5b69986b44df580b54b3ba3ab29e5f84d3ffeac3e179c462e6be4ca04887a4',
    14: '6f6647a95f4123516248908c38d8f042d954f2ef65f676727f70f9af34651479',
    15: 'ed0f57bf8174faa91e7a8d52b4059de372a00c617d8f56caff2bea28ffc5f2e2',
}
RUST_SCORES = {
    11: 'abffed02231c2951bd2ec5db86cbad964fde37e8a3df4a781df5a22f86bf8738',
    12: '118789f5d3ca4bca19ab0feb27fa4ffc3b28f5ba434853a8493c9fd9840f7be5',
    13: 'a71b9c154fb714fdacbd814a30d44d79c93bd406fa6320f9cf0420a37676009c',
    14: 'ffcf50e0952859d7720af6f448df7536476df3d195d0695f8ec20a95ffc3df30',
    15: '9a391c4e2d9120aee37ae62ed03520d1a04a700f558c3a7e37d206537fe31dd9',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_dev_protocol.py').resolve()]
    values = {'python': [], 'rust': []}
    for kind, capture_hashes, score_hashes in [('python', CAPTURES, SCORES), ('rust', RUST_CAPTURES, RUST_SCORES)]:
        for seed in range(11, 16):
            prefix = 'rust-' if kind == 'rust' else ''
            capture = ROOT / f'reports/phase4-lm1b-small-{prefix}dev-seed-{seed}-v1.json'
            score = ROOT / f'reports/phase4-lm1b-small-{prefix}dev-score-{seed}-v1.json'
            if digest_file(capture) != capture_hashes[seed] or digest_file(score) != score_hashes[seed]:
                raise ValueError('capture or score hash differs')
            item, metric = json.loads(capture.read_bytes()), json.loads(score.read_bytes())
            if ((item['size'], item['seed'], item['count'], item['batch_size']) != ('small', seed, 256, 8)
                    or metric['count'] != 256 or metric['upstream_exact'] is not True
                    or metric['sample_sha256'] != {'development': capture_hashes[seed]}
                    or len(metric['entropy']) != 256):
                raise ValueError('capture or scorer metadata differs')
            if kind == 'rust' and item['artifact_manifest_sha256'] != '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76':
                raise ValueError('Rust artifact identity differs')
            ppl, entropy = metric['ppl'], statistics.mean(metric['entropy'])
            if not math.isfinite(ppl) or not math.isfinite(entropy):
                raise ValueError('nonfinite metric')
            values[kind].append({'seed': seed, 'ppl': ppl, 'entropy': entropy})
            watched.extend([capture, score])
    before = {str(path): digest_file(path) for path in watched}
    means = {kind: {key: statistics.mean(row[key] for row in rows) for key in ('ppl', 'entropy')} for kind, rows in values.items()}
    delta = {key: means['rust'][key] - means['python'][key] for key in ('ppl', 'entropy')}
    rng = random.Random(42)
    draws = {key: [] for key in delta}
    for _ in range(10000):
        a = [rng.randrange(5) for _ in range(5)]
        b = [rng.randrange(5) for _ in range(5)]
        for key in draws:
            draws[key].append(statistics.mean(values['rust'][i][key] for i in b) - statistics.mean(values['python'][i][key] for i in a))
    intervals = {key: [sorted(items)[249], sorted(items)[9749]] for key, items in draws.items()}
    result = {'scope': 'five non-held-out development seeds; descriptive only, not a quality gate',
              'values': values, 'means': means, 'rust_minus_python': delta,
              'independent_seed_bootstrap_95': intervals,
              'bootstrap': {'seed': 42, 'resamples': 10000, 'unit': 'seed mean', 'method': 'independent percentile'}}
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed during summary')
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    print(json.dumps({'means': means, 'difference': delta, 'intervals': intervals, 'sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
