"""Capture resumable, source-bound XSum Metal top-1 candidate output batches."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from capture_xsum_mlx_quality import ARTIFACT_SHA, DATA_SHA, canonical, digest, validate_output

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
BINARY = ROOT / 'target/release/ncm'
INPUT_SHA = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'
REFERENCE_SHA = 'b2a20dbad883417406387726d799c8f79ecf814507f58d01bd05204e16775125'


def read_batch(path, start, count):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('invalid batch file')
    value = json.loads(path.read_bytes())
    if type(value) is not dict or set(value) != {'record', 'sha256'}:
        raise ValueError('invalid batch envelope')
    record = value['record']
    if (type(record) is not dict or set(record) != {'start', 'count', 'result'} or
            type(record['start']) is not int or type(record['count']) is not int or
            record['start'] != start or record['count'] != count or
            hashlib.sha256(canonical(record)).hexdigest() != value['sha256']):
        raise ValueError('batch receipt differs')
    validate_output(record['result'], count)
    return record['result']


def publish(path, payload):
    temporary = path.with_name(path.name + f'.{os.getpid()}.part')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def checked_qualification(value, binary_sha):
    sources = value.get('source_sha256', {})
    if (not value.get('shapes_ok') or value.get('code_differences') != 0 or value.get('id_differences') != 0 or
            sources.get(str(BINARY)) != binary_sha or
            sources.get(str(ARTIFACT / 'manifest.json')) != ARTIFACT_SHA or
            sources.get(str(ROOT / 'reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json')) != INPUT_SHA or
            sources.get(str(ROOT / 'reports/phase4-xsum-mlx-cli-inputs-v1/reference.json')) != REFERENCE_SHA):
        raise ValueError('candidate qualification differs')


def validate_batch_sequence(files, starts):
    if [p.name for p in files] != [f'batch-{start:05}.json' for start in starts[:len(files)]]:
        raise ValueError('batch sequence has gaps or extra files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--stop', type=int, default=11334)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon required')
    if not 1 <= args.stop <= 11334:
        raise ValueError('invalid stop')
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
            digest(BINARY) != args.binary_sha256 or args.qualification.is_symlink() or
            digest(args.qualification) != args.qualification_sha256):
        raise ValueError('input, executable or qualification differs')
    checked_qualification(json.loads(args.qualification.read_bytes()), args.binary_sha256)
    data = json.loads(DATA.read_bytes())
    if (data.get('profile') != 'xsum' or len(data.get('inputs', [])) != 11334 or
            any(type(row) is not list or len(row) != 560 or any(type(x) is not int or not 0 <= x < 50304
                                                                for x in row) for row in data['inputs'])):
        raise ValueError('invalid input dataset')
    watched = [DATA, BINARY, args.qualification.resolve(), Path(__file__).resolve(),
               ROOT / 'tools/bench/capture_xsum_mlx_quality.py', ROOT / 'Cargo.toml', ROOT / 'Cargo.lock',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    manifest = {'head': head, 'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA,
                'binary_sha256': args.binary_sha256, 'qualification_sha256': args.qualification_sha256,
                'source_sha256': before, 'batch_size': 32, 'rows': args.stop,
                'scope': 'top-1 CFG-zero Metal CLI; quality compared separately'}
    if output.exists():
        if not output.is_dir() or (output / 'manifest.json').is_symlink() or json.loads((output / 'manifest.json').read_bytes()) != manifest:
            raise ValueError('existing output identity differs')
    else:
        output.mkdir(parents=True)
        publish(output / 'manifest.json', canonical(manifest) + b'\n')
    starts = list(range(0, args.stop, 32))
    files = sorted(output.glob('batch-*.json'))
    validate_batch_sequence(files, starts)
    for start in starts:
        rows = data['inputs'][start:min(start + 32, args.stop)]
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in rows]
        input_path, batch_path = output / f'input-{start:05}.json', output / f'batch-{start:05}.json'
        expected = canonical(prefixes) + b'\n'
        if input_path.exists():
            if input_path.is_symlink() or input_path.read_bytes() != expected:
                raise ValueError('existing batch input differs')
        else:
            publish(input_path, expected)
        if batch_path.exists():
            read_batch(batch_path, start, len(rows))
            continue
        command = [str(BINARY), 'sample-xsum', str(ARTIFACT), 'mlx', str(input_path)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300, check=False)
        attempt = output / f'stderr-{start:05}-{os.getpid()}.txt'
        with attempt.open('xb') as stream:
            stream.write(process.stderr)
        if process.returncode or len(process.stdout) > 4 * 1024 * 1024:
            raise RuntimeError(f'CLI failed or returned too much data at row {start}')
        result = json.loads(process.stdout)
        validate_output(result, len(rows))
        record = {'start': start, 'count': len(rows), 'result': result}
        envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
        publish(batch_path, canonical(envelope) + b'\n')
        read_batch(batch_path, start, len(rows))
        print(f'{start + len(rows)}/{args.stop}', flush=True)
    if (before != {str(p): digest(p) for p in watched} or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('source or input changed during capture')
    complete = output / 'complete.json'
    expected_complete = {'rows': args.stop, 'binary_sha256': args.binary_sha256}
    if complete.exists():
        if complete.is_symlink() or json.loads(complete.read_bytes()) != expected_complete:
            raise ValueError('completion receipt differs')
    else:
        publish(complete, canonical(expected_complete) + b'\n')
    print('capture complete', flush=True)


if __name__ == '__main__':
    main()
