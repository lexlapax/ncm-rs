"""Compare complete XSum Metal candidate output to the retained Metal baseline."""
import argparse
import json
from pathlib import Path

from capture_xsum_mlx_candidate import read_batch
from capture_xsum_mlx_quality import ARTIFACT_SHA, DATA_SHA, digest, validate_output

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'reports/phase4-xsum-mlx-full-top1-captured-v1'
BASELINE_SHA = 'b0e6efa956268e3e424ac5a37a11f60b860bcedf8b2c1ac7661a2f4dc4d91923'


def compare_batch(reference, candidate, start):
    validate_output(reference, len(reference['codes']))
    validate_output(candidate, len(reference['codes']))
    result = {'rows': len(reference['codes']), 'code_differences': 0, 'id_differences': 0,
              'code_rows': [], 'id_rows': []}
    for offset, (a, b, c, d) in enumerate(zip(reference['codes'], candidate['codes'], reference['ids'], candidate['ids'])):
        code_diff = sum(x != y for x, y in zip(a, b))
        id_diff = sum(x != y for x, y in zip(c, d))
        result['code_differences'] += code_diff
        result['id_differences'] += id_diff
        if code_diff:
            result['code_rows'].append(start + offset)
        if id_diff:
            result['id_rows'].append(start + offset)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-manifest-sha256', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    baseline_manifest, candidate_manifest = BASELINE / 'manifest.json', args.candidate / 'manifest.json'
    if digest(baseline_manifest) != BASELINE_SHA or digest(candidate_manifest) != args.candidate_manifest_sha256:
        raise ValueError('capture manifest differs')
    baseline = json.loads(baseline_manifest.read_bytes())
    candidate = json.loads(candidate_manifest.read_bytes())
    expected = {'rows': 11334, 'batch_size': 32, 'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA}
    if (any(baseline.get(k) != v or candidate.get(k) != v for k, v in expected.items()) or
            candidate.get('head') != args.head or candidate.get('binary_sha256') != args.binary_sha256):
        raise ValueError('capture identity differs')
    if (json.loads((BASELINE / 'complete.json').read_bytes()) !=
            {'rows': 11334, 'binary_sha256': baseline['binary_sha256']} or
            json.loads((args.candidate / 'complete.json').read_bytes()) !=
            {'rows': 11334, 'binary_sha256': args.binary_sha256}):
        raise ValueError('capture is incomplete')
    starts = list(range(0, 11334, 32))
    for directory in (BASELINE, args.candidate):
        if [p.name for p in sorted(directory.glob('batch-*.json'))] != [f'batch-{i:05}.json' for i in starts]:
            raise ValueError('batch sequence differs')
    result = {'rows': 0, 'code_differences': 0, 'id_differences': 0, 'code_rows': [], 'id_rows': [],
              'baseline_manifest_sha256': BASELINE_SHA, 'candidate_manifest_sha256': args.candidate_manifest_sha256,
              'scope': 'exact candidate-to-retained Metal outputs; does not waive the Python raw-ID gate'}
    for start in starts:
        count = min(32, 11334 - start)
        ref = read_batch(BASELINE / f'batch-{start:05}.json', start, count)
        actual = read_batch(args.candidate / f'batch-{start:05}.json', start, count)
        batch = compare_batch(ref, actual, start)
        for key in ('rows', 'code_differences', 'id_differences'):
            result[key] += batch[key]
        for key in ('code_rows', 'id_rows'):
            result[key].extend(batch[key])
    if result['rows'] != 11334 or digest(baseline_manifest) != BASELINE_SHA or digest(candidate_manifest) != args.candidate_manifest_sha256:
        raise ValueError('coverage or manifest changed')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit('candidate differs from Metal baseline; result retained')


if __name__ == '__main__':
    main()
