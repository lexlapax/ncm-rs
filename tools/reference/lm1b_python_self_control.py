"""Descriptive frozen-comparator control between Python development and held-out seeds."""
import argparse
import json
import statistics
from pathlib import Path

from lm1b_dev_protocol import CAPTURES, SCORES
from lm1b_quality_compare import SEEDS, compare
from tinystories_artifact import ROOT, digest_file

HELDOUT_SUMMARY = '2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    heldout = ROOT / 'reports/phase4-lm1b-small-heldout-cuda-v1/summary.json'
    if digest_file(heldout) != HELDOUT_SUMMARY:
        raise ValueError('held-out summary identity differs')
    reference = json.loads(heldout.read_bytes())['records']['python']
    candidate, watched = [], [Path(__file__).resolve(), heldout, Path(__file__).with_name('lm1b_quality_compare.py').resolve(), Path(__file__).with_name('lm1b_dev_protocol.py').resolve()]
    for development_seed, label in zip(range(11, 16), SEEDS, strict=True):
        capture = ROOT / f'reports/phase4-lm1b-small-dev-seed-{development_seed}-v1.json'
        score = ROOT / f'reports/phase4-lm1b-small-dev-score-{development_seed}-v1.json'
        if digest_file(capture) != CAPTURES[development_seed] or digest_file(score) != SCORES[development_seed]:
            raise ValueError('development identity differs')
        item, metric = json.loads(capture.read_bytes()), json.loads(score.read_bytes())
        if ((item['size'], item['seed'], item['count'], item['batch_size']) != ('small', development_seed, 256, 8)
                or metric.get('count') != 256 or metric.get('upstream_exact') is not True
                or metric.get('sample_sha256') != {'development': CAPTURES[development_seed]}
                or len(metric.get('entropy', [])) != 256):
            raise ValueError('development metadata differs')
        candidate.append({'seed': label, 'count': 256, 'ppl': metric['ppl'], 'entropy': statistics.mean(metric['entropy'])})
        watched.extend([capture, score])
    before = {str(path): digest_file(path) for path in watched}
    comparison = compare(reference, candidate)
    result = {'scope': 'descriptive Python-versus-Python independent seed control; not a candidate gate or new margin',
              'heldout_seeds': list(SEEDS), 'development_seeds': list(range(11, 16)),
              'development_labels_only_for_comparator': list(SEEDS), 'comparison': comparison,
              'heldout_summary_sha256': HELDOUT_SUMMARY}
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed')
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    print(json.dumps({'comparison': comparison, 'sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
