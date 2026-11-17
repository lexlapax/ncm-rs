"""Capture the explicit WMT14 prefix-cache candidate against retained Python outputs."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / 'reports/phase4-wmt14-tokenizer-reference-v1.json'
ARTIFACT = ROOT / 'artifacts/canonical/wmt14-inference-v1-final'
PROTOCOL = ROOT / 'docs/results/2026-11-16-phase4-wmt14/heldout-protocol-v1.md'
PREFIX_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'
ARTIFACT_SHA = '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81'
REFERENCES = {
    ('base', 35): ('reports/phase4-wmt14-base-supervisor-smoke-v1/python.json', 'cea59ae16487d418e06013696bc54fc4985ef34cd243cd17f4920b5b93dc61f3'),
    ('cfg1', 35): ('reports/phase4-wmt14-cfg-supervisor-smoke-v1/python.json', '10d009a36f2a2282aa2d7bf1a530f60aa0b1b5f9eb58898069d0a3ddd3cea82f'),
    ('base', 3003): ('reports/phase4-wmt14-base-cuda-full-v1/python.json', 'e287a9c784d927cc319572fed6679bbf4cd9ffc1489c0f15028f71e13c47bce0'),
    ('cfg1', 3003): ('reports/phase4-wmt14-cfg-cuda-full-v1/python.json', '78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    return head, dirty


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cfg1'], required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--rows', type=int, choices=[35, 3003], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if args.output.is_symlink() or output.exists() or not output.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output directory under reports required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_VISIBLE_DEVICES',
                                      'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('unsupported environment override')
    reference_rel, reference_sha = REFERENCES[(args.mode, args.rows)]
    reference = ROOT / reference_rel
    binary = ROOT / 'target/release/examples' / 'wmt14_cfg_recompute_capture'
    comparator = ROOT / 'tools/reference' / ('compare_wmt14_cuda_capture.py' if args.mode == 'base' else 'compare_wmt14_cfg_capture.py')
    cudnn = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9'
    libraries = sorted(cudnn.parent.glob('libcudnn*.so.9'))
    if cudnn not in libraries:
        raise ValueError('cuDNN library inventory is missing')
    watched = [Path(__file__).resolve(), binary, comparator, PREFIX, ARTIFACT / 'manifest.json', PROTOCOL, reference, *libraries]
    before = {str(path.relative_to(ROOT)): digest(path) for path in watched}
    if (source_state() != (args.head, '') or before[str(PREFIX.relative_to(ROOT))] != PREFIX_SHA or
            before[str((ARTIFACT / 'manifest.json').relative_to(ROOT))] != ARTIFACT_SHA or
            before[str(PROTOCOL.relative_to(ROOT))] != args.protocol_sha256 or
            before[str(binary.relative_to(ROOT))] != args.binary_sha256 or before[reference_rel] != reference_sha):
        raise ValueError('source, binary, artifact or reference identity differs')
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', {'mode': args.mode, 'rows': args.rows, 'head': args.head,
               'binary_sha256': args.binary_sha256, 'protocol_sha256': args.protocol_sha256,
               'reference_sha256': reference_sha, 'watched': before,
               'scope': 'explicit near-tie recomputation diagnostic retest on previously inspected data; quality and performance are separate'})
    env = dict(os.environ, OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', LD_LIBRARY_PATH=str(cudnn.parent))
    command = [str(binary), str(ARTIFACT), str(PREFIX), str(output / 'rust.json'), '0', str(args.rows)]
    with (output / 'rust.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'candidate capture failed with exit {result.returncode}; output retained')
    rust_sha = digest(output / 'rust.json')
    compare = [str(ROOT / '.reference/venv/bin/python'), str(comparator), str(reference), str(output / 'rust.json'),
               str(output / 'comparison.json'), '--reference-sha256', reference_sha, '--candidate-sha256', rust_sha]
    with (output / 'comparison.log').open('x') as log:
        result = subprocess.run(compare, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode not in (0, 1) or not (output / 'comparison.json').is_file():
        raise RuntimeError('comparison failed without a result; output retained')
    if source_state() != (args.head, '') or before != {str(path.relative_to(ROOT)): digest(path) for path in watched}:
        raise ValueError('source or input changed during candidate capture')
    write_json(output / 'complete.json', {'rust_sha256': rust_sha, 'comparison_sha256': digest(output / 'comparison.json'),
               'comparison_exit': result.returncode})
    print(f'candidate capture complete; comparison exit {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
