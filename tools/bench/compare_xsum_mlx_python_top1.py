"""Compare complete Metal top-1 capture directly with retained pinned Python output."""
import argparse
import json
from pathlib import Path

from capture_xsum_mlx_quality import ARTIFACT_SHA, BINARY_SHA, DATA_SHA, digest, validate_output
from compare_xsum_mlx_quality import CAPTURE_SHA, envelope

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'reports/phase4-xsum-quality-python-full-rouge-v1.json'
PYTHON_SHA = '9e8c211611890833aab2cd4582ffbce13decdac26de92b53e5eab9f86fc3f76d'


def decoded_tokens(raw):
    """Match pinned evaluator's EOS truncation and EOS/PAD filtering before decode."""
    if 50256 in raw:
        raw = raw[:raw.index(50256)]
    return [token for token in raw if token not in (50256, 50257)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if digest(PYTHON) != PYTHON_SHA:
        raise ValueError('pinned Python scored output differs')
    reference = json.loads(PYTHON.read_bytes())
    if reference.get('start') != 0 or reference.get('stop') != 11334 or len(reference.get('rows', [])) != 11334:
        raise ValueError('unexpected Python corpus')
    manifest = json.loads((args.capture / 'manifest.json').read_bytes())
    expected = {'rows': 11334, 'batch_size': 32, 'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA,
                'binary_sha256': BINARY_SHA, 'script_sha256': CAPTURE_SHA,
                'head': 'e14ae02a517b0b3d1000d942f94d72bc408d851a'}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('unexpected Metal manifest')
    if json.loads((args.capture / 'complete.json').read_bytes()) != {'rows': 11334, 'binary_sha256': BINARY_SHA}:
        raise ValueError('Metal capture incomplete')
    result = {'rows': 11334, 'code_differences': 0, 'id_differences': 0,
              'code_rows': [], 'id_rows': [], 'decoded_token_rows': [],
              'reference_sha256': PYTHON_SHA, 'metal_manifest_sha256': digest(args.capture / 'manifest.json'),
              'scope': 'raw code and ID equality plus equality of IDs passed to the pinned text decoder'}
    for start in range(0, 11334, 32):
        record = envelope(args.capture / f'batch-{start:05}.json')
        length = min(32, 11334 - start)
        if record.get('start') != start or record.get('count') != length:
            raise ValueError('Metal batch identity differs')
        validate_output(record['result'], length)
        for offset, (codes, ids) in enumerate(zip(record['result']['codes'], record['result']['ids'])):
            index = start + offset
            old = reference['rows'][index]
            if old['index'] != index or len(old['codes']) != 200 or len(old['raw_ids']) != 48:
                raise ValueError('Python reference row identity differs')
            code_diff = sum(a != b for a, b in zip(codes, old['codes']))
            id_diff = sum(a != b for a, b in zip(ids, old['raw_ids']))
            result['code_differences'] += code_diff
            result['id_differences'] += id_diff
            if code_diff:
                result['code_rows'].append(index)
            if id_diff:
                result['id_rows'].append(index)
            if decoded_tokens(ids) != decoded_tokens(old['raw_ids']):
                result['decoded_token_rows'].append(index)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit('raw Metal and Python outputs differ; comparison retained')


if __name__ == '__main__':
    main()
