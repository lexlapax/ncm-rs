"""Summarize ten identity-bound LM1B-small held-out scorer reports."""
import argparse
import json
import math
import statistics
from pathlib import Path

from lm1b_quality_compare import SEEDS, compare
from tinystories_artifact import ROOT, UPSTREAM, digest_file

EVALUATOR_SHA = '6452cb5cc06a2355ab7cd16b9809b7144a557418e9041e4617f3c50aef5655fc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rust-binary-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    source = Path(__file__).with_name('lm1b_heldout_score.py').resolve()
    watched = [Path(__file__).resolve(), source, Path(__file__).with_name('lm1b_quality_compare.py').resolve()]
    scorer_sha = digest_file(source)
    records = {'python': [], 'rust': []}
    identities = {}
    for runtime in ('python', 'rust'):
        for seed in SEEDS:
            capture = args.directory / f'{runtime}-seed-{seed}.json'
            score = args.directory / f'{runtime}-seed-{seed}-score.json'
            capture_sha, score_sha = digest_file(capture), digest_file(score)
            item, metric = json.loads(capture.read_bytes()), json.loads(score.read_bytes())
            if ((item['size'], item['seed'], item['count'], item['batch_size']) != ('small', seed, 256, 8)
                    or metric.get('runtime') != runtime or metric.get('seed') != seed or metric.get('count') != 256
                    or metric.get('sample_sha256') != {'heldout': capture_sha} or metric.get('upstream') != UPSTREAM
                    or metric.get('evaluator_sha256') != EVALUATOR_SHA or metric.get('scorer_sha256') != scorer_sha
                    or metric.get('upstream_exact') is not True or len(metric.get('entropy', [])) != 256):
                raise ValueError('capture or scorer identity differs')
            if runtime == 'rust' and item.get('binary_sha256') != args.rust_binary_sha256:
                raise ValueError('Rust binary identity differs')
            entropy, ppl = statistics.mean(metric['entropy']), metric['ppl']
            if not math.isfinite(ppl) or not math.isfinite(entropy):
                raise ValueError('nonfinite metric')
            records[runtime].append({'seed': seed, 'count': 256, 'ppl': ppl, 'entropy': entropy})
            identities[f'{runtime}:{seed}'] = {'capture_sha256': capture_sha, 'score_sha256': score_sha}
            watched.extend([capture, score])
    before = {str(path): digest_file(path) for path in watched}
    result = {'scope': 'held-out metric comparison only; full runtime, hardware and performance gates are separate',
              'identities': identities, 'records': records, 'comparison': compare(records['python'], records['rust']),
              'scorer_sha256': scorer_sha, 'rust_binary_sha256': args.rust_binary_sha256}
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed during summary')
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    print(json.dumps({'comparison': result['comparison'], 'sha256': digest_file(args.output)}, sort_keys=True))
    if not result['comparison']['metrics_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
