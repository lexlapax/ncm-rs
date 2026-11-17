"""Capture resumable candidate Metal stochastic outputs under the frozen protocol."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from capture_xsum_mlx_candidate import checked_qualification, publish
from capture_xsum_mlx_quality import ARTIFACT_SHA, DATA_SHA, canonical, digest, validate_output

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
BINARY = ROOT / 'target/release/ncm'
PROTOCOL = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-followup-protocol-v1.json'
PROTOCOL_SHA = '8ccd05295382f6ad33b17183a3f24e5d726da74a73fc1a7027eea81af10d5332'


def read_batch(path, seed, batch):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('invalid batch file')
    value = json.loads(path.read_bytes())
    if type(value) is not dict or set(value) != {'record', 'sha256'}:
        raise ValueError('invalid batch envelope')
    record = value['record']
    if (type(record) is not dict or set(record) != {'batch', 'start', 'batch_seed', 'result'} or
            any(type(record[k]) is not int for k in ('batch', 'start', 'batch_seed')) or
            record['batch'] != batch or record['start'] != 512 + 32 * batch or
            record['batch_seed'] != seed * 1000 + batch or
            hashlib.sha256(canonical(record)).hexdigest() != value['sha256']):
        raise ValueError('stochastic batch receipt differs')
    validate_output(record['result'], 32)
    return record['result']


def validate_sequence(files, count):
    if [p.name for p in files] != [f'batch-{i:02}.json' for i in range(len(files))] or len(files) > count:
        raise ValueError('batch sequence has gaps or extra files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=(4, 5, 6, 7, 8), required=True)
    parser.add_argument('--batches', type=int, default=64)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon required')
    if not 1 <= args.batches <= 64:
        raise ValueError('invalid batch count')
    output = args.output.resolve()
    if args.output.is_symlink() or not output.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be under reports')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH')):
        raise ValueError('dynamic library injection unsupported')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if (head != args.head or len(head) != 40 or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('checkout is not clean at the requested commit')
    if (DATA.is_symlink() or digest(DATA) != DATA_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA or
            digest(PROTOCOL) != PROTOCOL_SHA or digest(BINARY) != args.binary_sha256 or
            args.qualification.is_symlink() or digest(args.qualification) != args.qualification_sha256):
        raise ValueError('input, executable, protocol or qualification differs')
    checked_qualification(json.loads(args.qualification.read_bytes()), args.binary_sha256)
    protocol = json.loads(PROTOCOL.read_bytes())
    if protocol.get('heldout_rows') != [512, 2560] or protocol.get('heldout_seeds') != [4, 5, 6, 7, 8]:
        raise ValueError('unexpected frozen protocol')
    data = json.loads(DATA.read_bytes())
    if (data.get('profile') != 'xsum' or len(data.get('inputs', [])) != 11334 or
            any(type(row) is not list or len(row) != 560 or any(type(x) is not int or not 0 <= x < 50304
                                                                for x in row) for row in data['inputs'])):
        raise ValueError('invalid input dataset')
    watched = [DATA, BINARY, PROTOCOL, args.qualification.resolve(), Path(__file__).resolve(),
               ROOT / 'tools/bench/capture_xsum_mlx_candidate.py', ROOT / 'tools/bench/capture_xsum_mlx_quality.py',
               ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / '.reference/mlx-native/lib/libmlx.dylib',
               ROOT / '.reference/mlx-native/lib/mlx.metallib']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    manifest = {'head': head, 'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA,
                'binary_sha256': args.binary_sha256, 'protocol_sha256': PROTOCOL_SHA,
                'qualification_sha256': args.qualification_sha256, 'source_sha256': before,
                'seed': args.seed, 'batches': args.batches, 'batch_size': 32, 'start': 512,
                'sampling': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0},
                'scope': 'candidate Metal stochastic output; comparison and scoring are separate'}
    if output.exists():
        if not output.is_dir() or (output / 'manifest.json').is_symlink() or json.loads((output / 'manifest.json').read_bytes()) != manifest:
            raise ValueError('existing output identity differs')
    else:
        output.mkdir(parents=True)
        publish(output / 'manifest.json', canonical(manifest) + b'\n')
    files = sorted(output.glob('batch-*.json'))
    validate_sequence(files, args.batches)
    for batch in range(args.batches):
        start = 512 + batch * 32
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in data['inputs'][start:start + 32]]
        input_path, batch_path = output / f'input-{batch:02}.json', output / f'batch-{batch:02}.json'
        expected = canonical(prefixes) + b'\n'
        if input_path.exists():
            if input_path.is_symlink() or input_path.read_bytes() != expected:
                raise ValueError('existing batch input differs')
        else:
            publish(input_path, expected)
        if batch_path.exists():
            read_batch(batch_path, args.seed, batch)
            continue
        batch_seed = args.seed * 1000 + batch
        command = [str(BINARY), 'sample-xsum', str(ARTIFACT), 'mlx', str(input_path), '0', str(batch_seed)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300, check=False)
        with (output / f'stderr-{batch:02}-{os.getpid()}.txt').open('xb') as stream:
            stream.write(process.stderr)
        if process.returncode or len(process.stdout) > 4 * 1024 * 1024:
            raise RuntimeError(f'CLI failed or returned too much data at batch {batch}')
        result = json.loads(process.stdout)
        validate_output(result, 32)
        record = {'batch': batch, 'start': start, 'batch_seed': batch_seed, 'result': result}
        publish(batch_path, canonical({'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}) + b'\n')
        read_batch(batch_path, args.seed, batch)
        print(f'seed {args.seed}: {batch + 1}/{args.batches}', flush=True)
    if (before != {str(p): digest(p) for p in watched} or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('source or input changed during capture')
    complete = output / 'complete.json'
    receipt = {'seed': args.seed, 'batches': args.batches, 'binary_sha256': args.binary_sha256}
    if complete.exists():
        if complete.is_symlink() or json.loads(complete.read_bytes()) != receipt:
            raise ValueError('completion receipt differs')
    else:
        publish(complete, canonical(receipt) + b'\n')
    print('capture complete', flush=True)


if __name__ == '__main__':
    main()
