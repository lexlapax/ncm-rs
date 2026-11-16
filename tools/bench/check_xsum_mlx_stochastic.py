"""Compare seeded XSum Metal CLI output with a 32-passage CUDA reference."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_SHA = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'
REF_SHA = 'f9e9aa42b6bcd4b8fd405e50c9689b11f0f3871bf1ec69c0535ef29b2d2c7f08'
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
    if reference.get('batch') != 32 or reference.get('seed') != 42 or reference.get('code_differences') != 0 or reference.get('id_differences') != 0:
        raise ValueError('unexpected reference')
    hashes = {str(p): digest(p) for p in files}
    output.mkdir(parents=True)
    command = [str(files[0]), 'sample-xsum', str(artifact), 'mlx', str(files[1]), '0', '42']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=300, check=False)
    (output / 'stdout.json').write_bytes(result.stdout)
    (output / 'stderr.txt').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Metal CLI exited {result.returncode}')
    actual = json.loads(result.stdout)
    if set(actual) != {'codes', 'ids'}:
        raise ValueError('unexpected CLI schema')
    code_diffs = sum(a != b for a, b in zip((v for row in actual['codes'] for v in row), reference['codes']))
    id_diffs = sum(a != b for a, b in zip((v for row in actual['ids'] for v in row), reference['ids']))
    shapes_ok = (len(actual['codes']) == len(actual['ids']) == 32
                 and all(len(row) == 200 for row in actual['codes'])
                 and all(len(row) == 48 for row in actual['ids']))
    if any(digest(p) != hashes[str(p)] for p in files):
        raise ValueError('input changed during comparison')
    for extra in (['1'],):
        rejected = subprocess.run(command[:5] + extra, cwd=ROOT, capture_output=True, timeout=30, check=False)
        if (rejected.returncode == 0 or rejected.stdout
                or b'only CFG 0' not in rejected.stderr):
            raise AssertionError('unsupported Metal sampling mode was not rejected explicitly')
    summary = {'schema_version': 1, 'command': command, 'source_sha256': hashes,
               'output_sha256': digest(output / 'stdout.json'), 'shapes_ok': shapes_ok,
               'code_differences': code_diffs, 'id_differences': id_diffs,
               'unsupported_positive_cfg_rejected': True,
               'scope': '32 real input passages; same SplitMix64 words, not stochastic quality or performance'}
    (output / 'result.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))
    if not shapes_ok or code_diffs or id_diffs:
        raise AssertionError('Metal seeded output differs from CUDA reference')


if __name__ == '__main__':
    main()
