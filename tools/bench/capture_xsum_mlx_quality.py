"""Capture top-1 Metal CLI outputs for pinned XSum passages, without Python inference.

Run from an already reviewed Mac checkout. Each batch launches the existing CLI;
receipts are retained under a new reports directory, and no model state is modified.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
BINARY = ROOT / 'target/release/ncm'
DATA_SHA = 'cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a'
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
BINARY_SHA = '498b70050edc69f7cc93255b9c681d3bc34ff628b0eb07b35f6e40223654cac2'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def validate_output(value, count):
    if type(value) is not dict or set(value) != {'codes', 'ids'}:
        raise ValueError('unexpected CLI output schema')
    for key, width, limit in (('codes', 200, 4096), ('ids', 48, 50304)):
        rows = value[key]
        if (type(rows) is not list or len(rows) != count or any(
            type(row) is not list or len(row) != width or any(
                type(x) is not int or not 0 <= x < limit for x in row) for row in rows)):
            raise ValueError(f'invalid {key} rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop', type=int, default=11334)
    args = parser.parse_args()
    if not 1 <= args.stop <= 11334:
        raise ValueError('invalid stop')
    output = args.output.resolve()
    if (args.output.exists() or args.output.is_symlink()
            or not output.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('new output directory under reports required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH')):
        raise ValueError('dynamic library injection unsupported')
    if digest(DATA) != DATA_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA:
        raise ValueError('input or artifact differs')
    data = json.loads(DATA.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('inputs', [])) != 11334:
        raise ValueError('unexpected XSum input dataset')
    if any(type(row) is not list or len(row) != 560 or any(
        type(x) is not int or not 0 <= x < 50304 for x in row) for row in data['inputs']):
        raise ValueError('invalid input rows')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != 'e14ae02a517b0b3d1000d942f94d72bc408d851a':
        raise ValueError('unexpected checked-out source')
    binary_sha = digest(BINARY)
    if binary_sha != BINARY_SHA:
        raise ValueError('binary differs from retained CLI screen')
    script_sha = digest(Path(__file__))
    output.mkdir()
    manifest = {'head': head, 'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA,
                'binary_sha256': binary_sha, 'script_sha256': script_sha, 'batch_size': 32, 'rows': args.stop,
                'scope': 'top-1 CFG-zero Metal CLI; output quality compared separately'}
    (output / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
    for start in range(0, args.stop, 32):
        rows = data['inputs'][start:min(start + 32, args.stop)]
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in rows]
        input_path = output / f'input-{start:05}.json'
        input_path.write_bytes(canonical(prefixes) + b'\n')
        command = [str(BINARY), 'sample-xsum', str(ARTIFACT), 'mlx', str(input_path)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300, check=False)
        (output / f'stderr-{start:05}.txt').write_bytes(run.stderr)
        if run.returncode:
            raise RuntimeError(f'CLI failed at row {start}: {run.returncode}')
        result = json.loads(run.stdout)
        validate_output(result, len(rows))
        record = {'start': start, 'count': len(rows), 'result': result}
        envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
        with (output / f'batch-{start:05}.json').open('xb') as stream:
            stream.write(canonical(envelope) + b'\n')
        print(f'{start + len(rows)}/{args.stop}', flush=True)
    if digest(DATA) != DATA_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA or digest(BINARY) != binary_sha:
        raise ValueError('input changed during capture')
    (output / 'complete.json').write_bytes(canonical({'rows': args.stop, 'binary_sha256': binary_sha}) + b'\n')


if __name__ == '__main__':
    main()
