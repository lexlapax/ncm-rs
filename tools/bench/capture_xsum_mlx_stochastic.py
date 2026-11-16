"""Capture the predeclared Metal stochastic XSum comparison for one seed."""
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
PROTOCOL = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-followup-protocol-v1.json'
DATA_SHA = 'cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a'
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
BINARY_SHA = '498b70050edc69f7cc93255b9c681d3bc34ff628b0eb07b35f6e40223654cac2'
PROTOCOL_SHA = '8ccd05295382f6ad33b17183a3f24e5d726da74a73fc1a7027eea81af10d5332'
HEAD = 'e14ae02a517b0b3d1000d942f94d72bc408d851a'


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
    parser.add_argument('--seed', type=int, choices=(4, 5, 6, 7, 8), required=True)
    parser.add_argument('--batches', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.batches <= 64:
        raise ValueError('invalid batch count')
    output = args.output.resolve()
    if (args.output.exists() or args.output.is_symlink()
            or not output.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('new output directory under reports required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH')):
        raise ValueError('dynamic library injection unsupported')
    watched = {DATA: DATA_SHA, ARTIFACT / 'manifest.json': ARTIFACT_SHA, BINARY: BINARY_SHA,
               PROTOCOL: PROTOCOL_SHA}
    if any(digest(path) != sha for path, sha in watched.items()):
        raise ValueError('input, binary, artifact or protocol differs')
    protocol = json.loads(PROTOCOL.read_bytes())
    if protocol.get('heldout_rows') != [512, 2560] or protocol.get('heldout_seeds') != [4, 5, 6, 7, 8]:
        raise ValueError('unexpected frozen protocol')
    data = json.loads(DATA.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('inputs', [])) != 11334:
        raise ValueError('unexpected XSum data')
    if any(type(row) is not list or len(row) != 560 or any(
        type(x) is not int or not 0 <= x < 50304 for x in row) for row in data['inputs']):
        raise ValueError('invalid input rows')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != HEAD:
        raise ValueError('unexpected checked-out source')
    script_sha = digest(Path(__file__))
    output.mkdir()
    manifest = {'head': head, 'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA,
                'binary_sha256': BINARY_SHA, 'protocol_sha256': PROTOCOL_SHA, 'script_sha256': script_sha,
                'seed': args.seed, 'batches': args.batches, 'batch_size': 32, 'start': 512,
                'sampling': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0},
                'scope': 'Metal stochastic CLI, independent stream; scoring and quality are separate'}
    (output / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
    for batch in range(args.batches):
        start = 512 + batch * 32
        rows = data['inputs'][start:start + 32]
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in rows]
        input_path = output / f'input-{batch:02}.json'
        input_path.write_bytes(canonical(prefixes) + b'\n')
        seed = args.seed * 1000 + batch
        command = [str(BINARY), 'sample-xsum', str(ARTIFACT), 'mlx', str(input_path), '0', str(seed)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300, check=False)
        (output / f'stderr-{batch:02}.txt').write_bytes(run.stderr)
        if run.returncode:
            raise RuntimeError(f'CLI failed at batch {batch}: {run.returncode}')
        result = json.loads(run.stdout)
        validate_output(result, 32)
        record = {'batch': batch, 'start': start, 'batch_seed': seed, 'result': result}
        envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
        with (output / f'batch-{batch:02}.json').open('xb') as stream:
            stream.write(canonical(envelope) + b'\n')
        print(f'seed {args.seed}: {batch + 1}/{args.batches}', flush=True)
    if any(digest(path) != sha for path, sha in watched.items()) or digest(Path(__file__)) != script_sha:
        raise ValueError('input changed during capture')
    (output / 'complete.json').write_bytes(canonical({'seed': args.seed, 'batches': args.batches,
                                                       'binary_sha256': BINARY_SHA}) + b'\n')


if __name__ == '__main__':
    main()
