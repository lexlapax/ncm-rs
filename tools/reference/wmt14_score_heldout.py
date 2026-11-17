"""Apply the frozen WMT14 paired held-out protocol to complete captures."""
import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from wmt14_score_dev import VERSIONS, digest, score


def paired(left, right, references, seed=42, repetitions=1000):
    if not left or len(left) != len(right) or len(left) != len(references):
        raise ValueError('invalid paired inputs')
    rng = random.Random(seed)
    metrics = ['bleu', 'bleu_lowercase', 'chrf', 'length_ratio']
    differences = {key: [] for key in metrics}
    identical = left == right
    for _ in range(repetitions):
        indices = [rng.randrange(len(left)) for _ in range(len(left))]
        if identical:
            for key in metrics:
                differences[key].append(0.0)
            continue
        ref = [references[i] for i in indices]
        before, _ = score([left[i] for i in indices], ref)
        after, _ = score([right[i] for i in indices], ref)
        for key in metrics:
            differences[key].append(after[key] - before[key])
    if repetitions != 1000:
        raise ValueError('protocol requires 1000 bootstrap samples')
    return {key: [sorted(values)[24], sorted(values)[974]] for key, values in differences.items()}, identical


def evaluate(left, right, references, mismatch_rows):
    before, signatures = score(left, references)
    after, signatures_after = score(right, references)
    if signatures != signatures_after:
        raise ValueError('metric signatures differ')
    intervals, identity_shortcut = paired(left, right, references)
    delta = {key: after[key] - before[key] for key in before}
    quality = all(delta[k] >= -0.5 and intervals[k][0] >= -0.5 for k in ('bleu', 'bleu_lowercase', 'chrf'))
    length = abs(delta['length_ratio']) <= 0.01 and all(abs(x) <= 0.01 for x in intervals['length_ratio'])
    return {'python': before, 'rust': after, 'delta': delta, 'paired_intervals_95': intervals,
            'signatures': signatures, 'raw_id_mismatch_rows': mismatch_rows,
            'all_passed': not mismatch_rows and quality and length,
            'bootstrap_identical_hypotheses_shortcut': identity_shortcut}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference_text', type=Path)
    p.add_argument('candidate_text', type=Path)
    p.add_argument('comparison', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--reference-text-sha256', required=True)
    p.add_argument('--candidate-text-sha256', required=True)
    p.add_argument('--comparison-sha256', required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--mode', choices=['base', 'cfg1'], required=True)
    args = p.parse_args()
    protocol = Path('docs/results/2026-11-16-phase4-wmt14/heldout-protocol-v1.md')
    watched = [(args.reference_text, args.reference_text_sha256), (args.candidate_text, args.candidate_text_sha256),
               (args.comparison, args.comparison_sha256), (protocol, args.protocol_sha256)]
    if args.output.exists() or args.output.is_symlink() or any(digest(path) != sha for path, sha in watched):
        raise ValueError('output exists or input identity differs')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('scorer dependency differs')
    a, b, comparison = (json.loads(path.read_bytes()) for path in (args.reference_text, args.candidate_text, args.comparison))
    if (a['start'], a['count'], b['start'], b['count']) != (0, 3003, 0, 3003) or a['references'] != b['references']:
        raise ValueError('complete matching reference rows required')
    if (len(a['hypotheses']) != 3003 or len(b['hypotheses']) != 3003 or len(a['references']) != 3003 or
            a['dataset_sha256'] != '36035390ce542834f8dfeeac004c1b532aff247a4ff39910f4a16efd009ca47c' or
            b['dataset_sha256'] != a['dataset_sha256'] or
            a['tokenizer_manifest_sha256'] != 'e1f881e225d651271d01d4067b3308578eb836703fdd25cfb17e7a1b95fcd69a' or
            b['tokenizer_manifest_sha256'] != a['tokenizer_manifest_sha256']):
        raise ValueError('dataset, tokenizer or row coverage differs')
    if (comparison['start'], comparison['count']) != (0, 3003) or comparison['reference_sha256'] != a['capture_sha256'] or comparison['candidate_sha256'] != b['capture_sha256']:
        raise ValueError('output comparison identity differs')
    if (args.mode == 'cfg1') != (comparison.get('cfg_scale', 0.0) == 1.0):
        raise ValueError('comparison mode differs')
    if any(type(row) is not int or not 0 <= row < 3003 for row in comparison['id_rows'] + comparison['code_rows']) or any(
            sorted(set(comparison[key])) != comparison[key] for key in ('id_rows', 'code_rows')):
        raise ValueError('invalid difference row')
    full = evaluate(a['hypotheses'], b['hypotheses'], a['references'], comparison['id_rows'])
    heldout = evaluate(a['hypotheses'][128:], b['hypotheses'][128:], a['references'][128:],
                       [row for row in comparison['id_rows'] if row >= 128])
    if any(digest(path) != sha for path, sha in watched):
        raise ValueError('input changed during scoring')
    result = {'scope': 'frozen WMT14 held-out comparison and full-set context', 'mode': args.mode,
              'protocol_sha256': args.protocol_sha256, 'versions': VERSIONS, 'bootstrap_seed': 42,
              'bootstrap_resamples': 1000, 'code_difference_rows': comparison['code_rows'],
              'full': full, 'heldout': heldout}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'mode': args.mode, 'heldout_passed': heldout['all_passed'], 'full_passed': full['all_passed']}, sort_keys=True))
    if not heldout['all_passed'] or not full['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
