"""Compare XSum Metal CLI output with a pinned 32-passage Python CUDA reference."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_SHA = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'
REF_SHA = 'b2a20dbad883417406387726d799c8f79ecf814507f58d01bd05204e16775125'
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
    output = args.output.resolve()
    if (args.output.is_symlink() or output.exists()
            or not output.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('new output under reports required')
    files = [args.binary.resolve(), args.inputs.resolve(), args.reference.resolve(),
             artifact / 'manifest.json', Path(__file__).resolve(), ROOT / 'crates/ncm-cli/src/main.rs']
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('regular input files required')
    if (digest(files[1]) != INPUT_SHA or digest(files[2]) != REF_SHA or digest(files[3]) != ARTIFACT_SHA):
        raise ValueError('input, reference or artifact differs')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH')):
        raise ValueError('dynamic library injection unsupported')
    reference = json.loads(files[2].read_bytes())
    if reference.get('batch') != 32 or reference.get('backend') != 'pinned Python CUDA':
        raise ValueError('unexpected reference')
    hashes = {str(p): digest(p) for p in files}
    output.mkdir(parents=True)
    command = [str(files[0]), 'sample-xsum', str(artifact), 'mlx', str(files[1])]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300, check=False)
    (output / 'stdout.json').write_bytes(result.stdout)
    (output / 'stderr.txt').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Metal CLI exited {result.returncode}')
    actual = json.loads(result.stdout)
    if set(actual) != {'codes', 'ids'}:
        raise ValueError('unexpected CLI schema')
    code_diffs = sum(a != b for ar, br in zip(actual['codes'], reference['codes']) for a, b in zip(ar, br))
    id_diffs = sum(a != b for ar, br in zip(actual['ids'], reference['ids']) for a, b in zip(ar, br))
    shapes_ok = (len(actual['codes']) == len(actual['ids']) == 32
                 and all(len(row) == 200 for row in actual['codes'])
                 and all(len(row) == 48 for row in actual['ids']))
    if any(digest(p) != hashes[str(p)] for p in files):
        raise ValueError('input changed during comparison')
    for extra in (['1'], ['1', '42']):
        rejected = subprocess.run(command + extra, cwd=ROOT, capture_output=True, timeout=30, check=False)
        if (rejected.returncode == 0 or rejected.stdout
                or b'only CFG 0' not in rejected.stderr):
            raise AssertionError('unsupported Metal sampling mode was not rejected explicitly')
    summary = {'schema_version': 1, 'command': command, 'source_sha256': hashes,
               'output_sha256': digest(output / 'stdout.json'), 'shapes_ok': shapes_ok,
               'code_differences': code_diffs, 'id_differences': id_diffs,
               'unsupported_positive_cfg_rejected': True,
               'scope': '32 real input passages; single correctness comparison, not performance'}
    (output / 'result.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))
    if not shapes_ok or code_diffs or id_diffs:
        raise AssertionError('Metal output differs from pinned CUDA reference')


if __name__ == '__main__':
    main()
