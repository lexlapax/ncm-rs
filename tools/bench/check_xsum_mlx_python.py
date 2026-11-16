"""Compare the independent Python MLX XSum adapter with pinned Python CUDA outputs."""
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
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    files = [args.inputs.resolve(), args.reference.resolve(), artifact / 'manifest.json',
             ROOT / 'tools/reference/phase4_xsum_mlx.py', ROOT / 'tools/reference/xsum_numpy_selection.py',
             ROOT / 'tools/reference/phase3_mlx_generator.py', ROOT / 'tools/reference/mlx_codec.py',
             ROOT / 'tools/reference/mlx_codec_assets.py', ROOT / 'tools/reference/mlx_operators.py', Path(__file__).resolve()]
    if (digest(files[0]) != INPUT_SHA or digest(files[1]) != REF_SHA or digest(files[2]) != ARTIFACT_SHA):
        raise ValueError('input, reference or artifact differs')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH')):
        raise ValueError('dynamic library injection unsupported')
    reference = json.loads(files[1].read_bytes())
    if reference.get('batch') != 32 or reference.get('backend') != 'pinned Python CUDA':
        raise ValueError('unexpected reference')
    hashes = {str(p): digest(p) for p in files}
    out.mkdir(parents=True)
    # Preserve the venv launcher path: resolving its symlink bypasses pyvenv.cfg.
    command = [str(args.python.absolute()), str(files[3]), '--artifact', str(artifact), '--inputs', str(files[0]),
               '--batch', '32', '--output', str(out / 'adapter.json')]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=600, check=False)
    (out / 'stdout.txt').write_bytes(result.stdout)
    (out / 'stderr.txt').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Python MLX adapter exited {result.returncode}')
    actual = json.loads((out / 'adapter.json').read_bytes())
    shapes = (actual.get('runtime') == 'python-mlx-adapted' and actual.get('batch') == 32
              and len(actual.get('codes', [])) == len(actual.get('ids', [])) == 32
              and all(len(row) == 200 for row in actual['codes']) and all(len(row) == 48 for row in actual['ids']))
    code_diff = sum(a != b for ar, br in zip(actual['codes'], reference['codes']) for a, b in zip(ar, br))
    id_diff = sum(a != b for ar, br in zip(actual['ids'], reference['ids']) for a, b in zip(ar, br))
    if any(digest(p) != hashes[str(p)] for p in files):
        raise ValueError('source changed during comparison')
    summary = {'schema_version': 1, 'scope': '32 real passages; initial Python MLX reference qualification',
               'command': command, 'source_sha256': hashes, 'adapter_sha256': digest(out / 'adapter.json'),
               'shapes_ok': shapes, 'code_differences': code_diff, 'id_differences': id_diff}
    (out / 'result.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))
    if not shapes or code_diff or id_diff:
        raise AssertionError('Python MLX adaptation differs from pinned CUDA reference')


if __name__ == '__main__':
    main()
