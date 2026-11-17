"""Compare candidate Metal stochastic outputs to the retained scored Metal captures."""
import argparse
import json
from pathlib import Path

from capture_xsum_mlx_quality import ARTIFACT_SHA, DATA_SHA, digest
from capture_xsum_mlx_stochastic_candidate import PROTOCOL_SHA, read_batch
from compare_xsum_mlx_candidate import compare_batch

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SHA = {
    4: '3435af78e37f1cc66342221c1475be6b413875359a873e18eb3cdb31ed8a0f17',
    5: 'aa27e14bb6a2e2ddee0b420a3b2d9bc4185efbb045dd038233bafbba549a8b81',
    6: '22a2f775655004597ebc8a3ef1269790421370c9e7bc092391d9dda538ade48c',
    7: 'fc433c4a3b565ea5c1f5aeaa786c92e1c402819ce28d7f9a869fc9a129f1f267',
    8: '6b55261c8de3ceb67c8c07abbf00c2ddd4cda9c8d6b425f949ae50d7ecd48824',
}
SAMPLING = {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0}


def compare_seed(baseline, candidate, seed):
    result = {'rows': 0, 'code_differences': 0, 'id_differences': 0, 'code_rows': [], 'id_rows': []}
    for batch in range(64):
        ref = read_batch(baseline / f'batch-{batch:02}.json', seed, batch)
        actual = read_batch(candidate / f'batch-{batch:02}.json', seed, batch)
        delta = compare_batch(ref, actual, 512 + 32 * batch)
        for key in ('rows', 'code_differences', 'id_differences'):
            result[key] += delta[key]
        for key in ('code_rows', 'id_rows'):
            result[key].extend(delta[key])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-template', required=True, help='path template containing {seed}')
    parser.add_argument('--manifest-sha256', nargs=5, required=True, help='candidate manifest hashes for seeds 4..8')
    parser.add_argument('--head', required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.candidate_template.count('{seed}') != 1:
        raise ValueError('new output and one seed placeholder required')
    result = {'seeds': {}, 'rows': 0, 'code_differences': 0, 'id_differences': 0,
              'scope': 'exact candidate-to-scored Metal stochastic outputs; not a fresh score or new held-out study'}
    manifests = []
    for seed, expected_candidate in zip(range(4, 9), args.manifest_sha256):
        baseline = ROOT / f'reports/phase4-xsum-mlx-stochastic-seed-{seed}-captured-v1'
        candidate = Path(args.candidate_template.format(seed=seed))
        bm, cm = baseline / 'manifest.json', candidate / 'manifest.json'
        if digest(bm) != BASELINE_SHA[seed] or digest(cm) != expected_candidate:
            raise ValueError('capture manifest differs')
        b, c = json.loads(bm.read_bytes()), json.loads(cm.read_bytes())
        expected = {'seed': seed, 'batches': 64, 'batch_size': 32, 'start': 512,
                    'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA,
                    'protocol_sha256': PROTOCOL_SHA, 'sampling': SAMPLING}
        if (any(b.get(k) != v or c.get(k) != v for k, v in expected.items()) or
                c.get('head') != args.head or c.get('binary_sha256') != args.binary_sha256):
            raise ValueError('capture identity differs')
        if (json.loads((baseline / 'complete.json').read_bytes()) !=
                {'seed': seed, 'batches': 64, 'binary_sha256': b['binary_sha256']} or
                json.loads((candidate / 'complete.json').read_bytes()) !=
                {'seed': seed, 'batches': 64, 'binary_sha256': args.binary_sha256}):
            raise ValueError('capture is incomplete')
        for directory in (baseline, candidate):
            if [p.name for p in sorted(directory.glob('batch-*.json'))] != [f'batch-{i:02}.json' for i in range(64)]:
                raise ValueError('batch sequence differs')
        delta = compare_seed(baseline, candidate, seed)
        if delta['rows'] != 2048:
            raise ValueError('seed coverage differs')
        delta.update({'baseline_manifest_sha256': BASELINE_SHA[seed], 'candidate_manifest_sha256': expected_candidate})
        result['seeds'][str(seed)] = delta
        for key in ('rows', 'code_differences', 'id_differences'):
            result[key] += delta[key]
        manifests.extend(((bm, BASELINE_SHA[seed]), (cm, expected_candidate)))
    if result['rows'] != 10240 or any(digest(path) != sha for path, sha in manifests):
        raise ValueError('coverage or manifest changed')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit('candidate differs from scored Metal baseline; result retained')


if __name__ == '__main__':
    main()
