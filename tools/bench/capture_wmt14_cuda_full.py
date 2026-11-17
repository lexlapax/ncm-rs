"""Run source-bound upstream and Rust WMT14 CUDA captures sequentially."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / 'reports/phase4-wmt14-tokenizer-reference-v1.json'
ARTIFACT = ROOT / 'artifacts/canonical/wmt14-inference-v1-final'
PREFIX_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'
ARTIFACT_SHA = '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81'
PROTOCOL = ROOT / 'docs/results/2026-11-16-phase4-wmt14/heldout-protocol-v1.md'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    return head, dirty


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['base', 'cfg1'], required=True)
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
    if source_state() != (args.head, '') or digest(PREFIX) != PREFIX_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA or digest(PROTOCOL) != args.protocol_sha256:
        raise ValueError('source, input or protocol identity differs')
    suffix = 'wmt14_cuda_capture' if args.mode == 'base' else 'wmt14_cfg_capture'
    binary_name = 'wmt14_capture' if args.mode == 'base' else 'wmt14_cfg_capture'
    comparator_name = 'compare_wmt14_cuda_capture' if args.mode == 'base' else 'compare_wmt14_cfg_capture'
    binary = ROOT / 'target/release/examples' / binary_name
    if digest(binary) != args.binary_sha256:
        raise ValueError('binary differs')
    sources = [Path(__file__).resolve(), ROOT / 'tools/reference' / f'{suffix}.py',
               ROOT / 'tools/reference' / f'{comparator_name}.py', ROOT / 'Cargo.lock']
    before = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    output.mkdir(parents=True)
    checked_json(output / 'manifest.json', {'mode': args.mode, 'rows': args.rows, 'batch_size': 4,
                 'head': args.head, 'binary_sha256': args.binary_sha256, 'protocol_sha256': args.protocol_sha256,
                 'prefix_sha256': PREFIX_SHA, 'artifact_sha256': ARTIFACT_SHA, 'sources': before,
                 'scope': 'sequential CUDA output capture; quality and performance are separate'})
    python = ROOT / '.reference/venv/bin/python'
    env = dict(os.environ, OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    commands = [
        [str(python), str(ROOT / 'tools/reference' / f'{suffix}.py'), '--upstream', str(ROOT / '.reference/upstream-ncm'),
         '--source', str(ROOT / 'artifacts/upstream/wmt14-phase4'), '--prefixes', str(PREFIX),
         '--output', str(output / 'python.json'), '--start', '0', '--count', str(args.rows)],
        [str(binary), str(ARTIFACT), str(PREFIX), str(output / 'rust.json'), '0', str(args.rows)],
    ]
    for label, command in zip(('python', 'rust'), commands, strict=True):
        with (output / f'{label}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'{label} capture failed with exit {result.returncode}; output retained')
        print(f'{label} capture complete', flush=True)
    python_sha, rust_sha = digest(output / 'python.json'), digest(output / 'rust.json')
    compare = [str(python), str(ROOT / 'tools/reference' / f'{comparator_name}.py'), str(output / 'python.json'),
               str(output / 'rust.json'), str(output / 'comparison.json'), '--reference-sha256', python_sha,
               '--candidate-sha256', rust_sha]
    with (output / 'comparison.log').open('x') as log:
        result = subprocess.run(compare, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode not in (0, 1) or not (output / 'comparison.json').is_file():
        raise RuntimeError('comparison failed without a result; output retained')
    if (source_state() != (args.head, '') or digest(PREFIX) != PREFIX_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA or
            digest(PROTOCOL) != args.protocol_sha256 or digest(binary) != args.binary_sha256 or
            before != {str(path.relative_to(ROOT)): digest(path) for path in sources}):
        raise ValueError('source or input changed during capture')
    checked_json(output / 'complete.json', {'python_sha256': python_sha, 'rust_sha256': rust_sha,
                 'comparison_sha256': digest(output / 'comparison.json'), 'comparison_exit': result.returncode})
    print(f'capture complete; comparison exit {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
