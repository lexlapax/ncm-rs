"""Compare a completed Metal XSum capture with the retained full CUDA capture."""
import argparse
import hashlib
import json
from pathlib import Path

from capture_xsum_mlx_quality import ARTIFACT_SHA, BINARY_SHA, DATA_SHA, canonical, digest, validate_output

ROOT = Path(__file__).resolve().parents[2]
CUDA_MANIFESTS = ('ae580e8ac25b59c2b782f3aeade72daf24ffbe8f4bd637d882f671607bfb2240',
                  '393be26641af3531fef4d14d5129a314e090c52fbe92a9cb2202d9edf448bd2f')
CAPTURE_SHA = '5c6c463008b5bf5ca6d8993dff1808375f205919766888afe861c6230acb081f'


def envelope(path):
    value = json.loads(path.read_bytes())
    if (type(value) is not dict or set(value) != {'record', 'sha256'} or
            value['sha256'] != hashlib.sha256(canonical(value['record'])).hexdigest()):
        raise ValueError(f'invalid receipt: {path}')
    return value['record']


def compare(metal_rows, cuda_rows):
    if len(metal_rows) != len(cuda_rows):
        raise ValueError('row counts differ')
    summary = {'rows': len(metal_rows), 'code_differences': 0, 'id_differences': 0,
               'code_rows': [], 'id_rows': []}
    for index, (metal, cuda) in enumerate(zip(metal_rows, cuda_rows)):
        for key, label in (('codes', 'code'), ('ids', 'id')):
            n = sum(a != b for a, b in zip(metal[key], cuda[key]))
            summary[f'{label}_differences'] += n
            if n:
                summary[f'{label}_rows'].append(index)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    manifest = json.loads((args.capture / 'manifest.json').read_bytes())
    count = manifest.get('rows')
    if (type(count) is not int or not 1 <= count <= 11334 or manifest.get('batch_size') != 32
            or manifest.get('data_sha256') != DATA_SHA or manifest.get('artifact_sha256') != ARTIFACT_SHA
            or manifest.get('binary_sha256') != BINARY_SHA or manifest.get('script_sha256') != CAPTURE_SHA
            or manifest.get('head') != 'e14ae02a517b0b3d1000d942f94d72bc408d851a'):
        raise ValueError('unexpected Metal manifest')
    complete = json.loads((args.capture / 'complete.json').read_bytes())
    if complete != {'rows': count, 'binary_sha256': BINARY_SHA}:
        raise ValueError('capture not complete')
    metal = []
    for start in range(0, count, 32):
        record = envelope(args.capture / f'batch-{start:05}.json')
        length = min(32, count - start)
        if record.get('start') != start or record.get('count') != length:
            raise ValueError('Metal batch identity differs')
        validate_output(record['result'], length)
        metal.extend({'codes': codes, 'ids': ids} for codes, ids in zip(
            record['result']['codes'], record['result']['ids']))
    cuda = [None] * count
    for shard, expected_hash in enumerate(CUDA_MANIFESTS):
        directory = ROOT / f'reports/phase4-xsum-reference-tie-full-v1-s{shard}'
        if digest(directory / 'manifest.json') != expected_hash:
            raise ValueError('CUDA reference manifest differs')
        indices = list(range(shard, 11334, 2))
        for ordinal, start in enumerate(range(0, len(indices), 32)):
            expected = indices[start:start + 32]
            if expected[0] >= count:
                break
            record = envelope(directory / f'batch-{ordinal:05}.json')
            if record.get('ordinal') != ordinal or record.get('indices') != expected:
                raise ValueError('CUDA batch identity differs')
            validate_output({'codes': record['codes'], 'ids': record['raw_ids']}, len(expected))
            for index, codes, ids in zip(expected, record['codes'], record['raw_ids']):
                if index < count:
                    if cuda[index] is not None:
                        raise ValueError('duplicate CUDA row')
                    cuda[index] = {'codes': codes, 'ids': ids}
    if any(row is None for row in cuda):
        raise ValueError('missing CUDA rows')
    result = compare(metal, cuda)
    result.update({'metal_manifest_sha256': digest(args.capture / 'manifest.json'),
                   'cuda_manifest_sha256': list(CUDA_MANIFESTS),
                   'scope': 'top-1 generated codes and IDs; equal IDs imply the same deterministic text and score only under the retained evaluator'})
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit('Metal and CUDA outputs differ; see the retained comparison')


if __name__ == '__main__':
    main()
