"""Compare CUDA and saved Metal top-1 for selected contiguous batches."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from capture_xsum_mlx_quality import DATA_SHA, canonical, digest, validate_output
from compare_xsum_mlx_quality import envelope
from compare_xsum_mlx_python_top1 import PYTHON, PYTHON_SHA

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / 'reports/phase4-xsum-mlx-full-top1-captured-v1'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
DATA = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
BINARY = ROOT / 'target/release/ncm'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
METAL_MANIFEST_SHA = 'b0e6efa956268e3e424ac5a37a11f60b860bcedf8b2c1ac7661a2f4dc4d91923'
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'


def differences(a, b, start):
    result = {'code_differences': 0, 'id_differences': 0, 'code_rows': [], 'id_rows': []}
    for offset, (ac, bc, ai, bi) in enumerate(zip(a['codes'], b['codes'], a['ids'], b['ids'])):
        cd = sum(x != y for x, y in zip(ac, bc))
        nd = sum(x != y for x, y in zip(ai, bi))
        result['code_differences'] += cd; result['id_differences'] += nd
        if cd:
            result['code_rows'].append(start + offset)
        if nd:
            result['id_rows'].append(start + offset)
    return result


def validate_starts(starts):
    if (not starts or len(set(starts)) != len(starts) or
            any(start < 0 or start % 32 or start + 32 > 11334 for start in starts)):
        raise ValueError('invalid contiguous batch starts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--starts', type=int, nargs='+', default=[4864, 4928])
    parser.add_argument('--cli-result', type=Path,
                        default=ROOT / 'reports/phase4-xsum-owned-loader-cli-v1/result.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.exists() or args.output.is_symlink() or not output.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                                      'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('environment overrides unsupported')
    validate_starts(args.starts)
    inputs = [DATA, PYTHON, CAPTURE / 'manifest.json', ARTIFACT / 'manifest.json', BINARY,
              Path(__file__).resolve(), args.cli_result.resolve()]
    hashes = {str(p): digest(p) for p in inputs}
    if (hashes[str(DATA)] != DATA_SHA or hashes[str(PYTHON)] != PYTHON_SHA
            or hashes[str(CAPTURE / 'manifest.json')] != METAL_MANIFEST_SHA
            or hashes[str(ARTIFACT / 'manifest.json')] != ARTIFACT_SHA):
        raise ValueError('input identity differs')
    cli_reference = json.loads(args.cli_result.read_bytes())
    if hashes[str(BINARY)] != cli_reference['source_sha256'][str(BINARY)]:
        raise ValueError('CUDA binary differs from prior CLI comparison')
    data, reference = json.loads(DATA.read_bytes()), json.loads(PYTHON.read_bytes())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'], text=True).strip()
    output.mkdir()
    results = {}
    for start in args.starts:
        path = CAPTURE / f'input-{start:05}.json'
        rows = data['inputs'][start:start + 32]
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in rows]
        if path.read_bytes() != canonical(prefixes) + b'\n':
            raise ValueError('captured input differs from dataset')
        metal = envelope(CAPTURE / f'batch-{start:05}.json')['result']
        validate_output(metal, 32)
        old = {'codes': [r['codes'] for r in reference['rows'][start:start + 32]],
               'ids': [r['raw_ids'] for r in reference['rows'][start:start + 32]]}
        validate_output(old, 32)
        env = os.environ.copy(); env['LD_LIBRARY_PATH'] = str(CUDNN); env['OMP_NUM_THREADS'] = '1'
        run = subprocess.run([str(BINARY), 'sample-xsum', str(ARTIFACT), 'cuda-cudnn', str(path)],
                             cwd=ROOT, env=env, capture_output=True, timeout=300, check=False)
        (output / f'cuda-{start}.stderr').write_bytes(run.stderr)
        (output / f'cuda-{start}.json').write_bytes(run.stdout)
        if run.returncode:
            raise RuntimeError(f'CUDA CLI failed: {run.returncode}')
        cuda = json.loads(run.stdout); validate_output(cuda, 32)
        results[str(start)] = {'cuda_vs_python': differences(cuda, old, start),
                               'metal_vs_python': differences(metal, old, start),
                               'metal_vs_cuda': differences(metal, cuda, start)}
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError('source or input changed during diagnostic')
    summary = {'source_sha256': hashes, 'gpu_identity_and_driver': gpu, 'batches': results,
               'scope': 'selected contiguous batches; saved Python used interleaved batches'}
    (output / 'result.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
