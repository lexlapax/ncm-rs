"""Capture source-bound stochastic XSum batches for the independent follow-up protocol."""
import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

from safetensors.torch import load_file

from xsum_artifact import INPUTS, UPSTREAM, digest_file, verify_input
from xsum_linked_libraries import linked_libraries
from xsum_quality_generate import DATA_SHA

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.reference/venv/bin/python'
SOURCE = ROOT / 'artifacts/upstream/xsum-phase4'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
BINARY = ROOT / 'target/release/ncm'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
TRACE = ROOT / 'tools/reference/xsum_cfg_trace.py'
DATA = ROOT / 'artifacts/canonical/text-phase4-v1/xsum.json'
SEEDS = {'development': (100, 101, 102, 103, 104), 'heldout': (4, 5, 6, 7, 8)}
METRICS = ('rouge1', 'rouge2', 'rougeL', 'generation_perplexity', 'mean_words')


def validate_protocol(protocol):
    margins = protocol.get('margins')
    if (protocol.get('schema_version') != 1 or protocol.get('status') != 'frozen'
            or protocol.get('development_seeds') != list(SEEDS['development'])
            or protocol.get('heldout_seeds') != list(SEEDS['heldout'])
            or protocol.get('data_sha256') != DATA_SHA
            or protocol.get('development_rows') != [0, 256]
            or protocol.get('original_protocol_sha256') != '2e9d5edd74770b2e3345198fef6c97318d37a3682111fd113bf0993cff594070'
            or protocol.get('heldout_rows') != [512, 2560]
            or protocol.get('bootstrap') != {'replicates': 10000, 'seed': 20261116, 'confidence': 0.975}
            or not isinstance(margins, dict) or set(margins) != set(METRICS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in margins.values())
            or len(protocol.get('development_score_sha256', [])) != 5):
        raise ValueError('protocol is not the frozen stochastic qualification')
    original_path = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-protocol-v1.json'
    if (digest_file(original_path) != protocol['original_protocol_sha256']
            or margins != json.loads(original_path.read_bytes())['margins']):
        raise ValueError('original frozen margins differ')


def sources():
    paths = [Path(__file__).resolve(), TRACE, ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', BINARY]
    for crate in ('ncm-core', 'ncm-model', 'ncm-io', 'ncm-tokenizers', 'ncm-cli', 'ncm-backend-cuda'):
        base = ROOT / 'crates' / crate
        paths.append(base / 'Cargo.toml')
        paths.extend(p for p in (base / 'src').rglob('*') if p.is_file())
    paths.append(ROOT / 'crates/ncm-backend-cuda/build.rs')
    return {str(path.relative_to(ROOT)): digest_file(path) for path in sorted(paths)}


def run(command, output, env, stderr_path=None):
    with contextlib.ExitStack() as stack:
        stdout = stack.enter_context(output.open('xb'))
        stderr = (stack.enter_context(stderr_path.open('xb')) if stderr_path is not None
                  else subprocess.STDOUT)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=300)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if code:
        raise RuntimeError(f'capture process failed: {command[0]} ({code})')


def valid_rows(rows, count, width, bound):
    return (type(rows) is list and len(rows) == count
            and all(type(row) is list and len(row) == width
                    and all(type(value) is int and 0 <= value < bound for value in row)
                    for row in rows))


def validate_batch(out, runtime, batch, seed, prefixes):
    if runtime == 'python':
        directory = out / f'reference-b{batch}'
        report = json.loads((directory / 'report.json').read_bytes())
        path = directory / 'trace.safetensors'
        if (report.get('upstream') != UPSTREAM or report.get('seed') != seed
                or report.get('batch') != 32 or report.get('cfg_scale') != 0.0
                or report.get('mode') != 'upstream temperature 1, top-k 0, top-p 1'
                or report.get('sources') != {name: spec[1] for name, spec in INPUTS.items()}
                or report.get('trace_sha256') != digest_file(path)):
            raise ValueError('Python trace provenance differs')
        trace = load_file(path)
        if trace['generator.prefix_ids'].tolist() != prefixes:
            raise ValueError('Python trace prefixes differ')
        codes = trace['generator.selected_codes'].tolist()
        ids = trace['generator.penalized_ids'].tolist()
    else:
        record = json.loads((out / f'rust-b{batch}.json').read_bytes())
        if set(record) != {'codes', 'ids'}:
            raise ValueError('unexpected Rust output schema')
        codes, ids = record['codes'], record['ids']
    if not valid_rows(codes, 32, 200, 4096) or not valid_rows(ids, 32, 48, 50304):
        raise ValueError('invalid sampled output')


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('capture interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=('python', 'rust'), required=True)
    parser.add_argument('--set', choices=tuple(SEEDS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if (args.seed not in SEEDS[args.set] or args.output.is_symlink()
            or not out.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('invalid seed or output')
    if any(os.environ.get(name) for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH',
           'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
           'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
           'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')):
        raise ValueError('environment overrides are unsupported')
    if digest_file(DATA) != DATA_SHA or digest_file(ARTIFACT / 'manifest.json') != ARTIFACT_SHA:
        raise ValueError('dataset or artifact identity differs')
    protocol_sha = None
    if args.set == 'development':
        raise ValueError('follow-up is held-out only')
    if args.set == 'heldout':
        if args.protocol is None or args.protocol.is_symlink() or not args.protocol.is_file():
            raise ValueError('frozen protocol required for held-out capture')
        protocol = json.loads(args.protocol.read_bytes())
        validate_protocol(protocol)
        protocol_sha = digest_file(args.protocol)
    elif args.protocol is not None:
        raise ValueError('development capture must not use a protocol')
    revision = subprocess.check_output(['git', '-C', str(ROOT / '.reference/upstream-ncm'),
                                        'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT / '.reference/upstream-ncm'),
                                     'status', '--porcelain'], text=True).strip()
    if revision != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(SOURCE / name, size, sha)
    source_hashes = sources()
    libraries = linked_libraries(BINARY, CUDNN)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1:
        raise ValueError('exactly one GPU required')
    manifest = {'schema_version': 1, 'runtime': args.runtime, 'set': args.set, 'seed': args.seed,
                'batch_seeds': [args.seed * 1000 + batch for batch in range(64)],
                'start': 0 if args.set == 'development' else 512, 'rows': 2048, 'batch_size': 32,
                'sampling': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0},
                'random_stream': 'PyTorch multinomial' if args.runtime == 'python' else 'SplitMix64 inverse-CDF',
                'data_sha256': DATA_SHA, 'artifact_sha256': ARTIFACT_SHA, 'upstream': revision,
                'checkpoint_sha256': {name: spec[1] for name, spec in INPUTS.items()},
                'source_sha256': source_hashes, 'native_libraries': libraries,
                'gpu_identity_and_driver': gpu[0], 'protocol_sha256': protocol_sha}
    if out.exists():
        metadata = out / 'manifest.json'
        if (out.is_symlink() or not out.is_dir() or metadata.is_symlink()
                or not metadata.is_file() or json.loads(metadata.read_bytes()) != manifest):
            raise ValueError('existing capture manifest differs')
    else:
        out.mkdir(parents=True)
        with (out / 'manifest.json').open('x') as file:
            json.dump(manifest, file, sort_keys=True, indent=2); file.write('\n')
    data = json.loads(DATA.read_bytes())
    env = os.environ.copy(); env['OMP_NUM_THREADS'] = '1'; env['TOKENIZERS_PARALLELISM'] = 'false'
    for batch in range(64):
        start = manifest['start'] + batch * 32
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in data['inputs'][start:start + 32]]
        input_path = out / f'input-b{batch}.json'
        encoded = json.dumps(prefixes, separators=(',', ':')).encode() + b'\n'
        if input_path.is_symlink():
            raise ValueError('input cannot be a symlink')
        if input_path.exists():
            if input_path.read_bytes() != encoded:
                raise ValueError('existing input differs')
        else:
            with input_path.open('xb') as file:
                file.write(encoded)
        result = out / (f'reference-b{batch}' if args.runtime == 'python' else f'rust-b{batch}.json')
        partial = out / (f'reference-b{batch}.part' if args.runtime == 'python' else f'rust-b{batch}.part')
        if result.is_symlink() or partial.exists() or partial.is_symlink():
            raise ValueError(f'partial batch requires inspection: {partial}')
        if not result.exists():
            seed = args.seed * 1000 + batch
            log = out / f'batch-{batch}.log'
            if args.runtime == 'python':
                command = [str(PYTHON), str(TRACE), '--upstream', str(ROOT / '.reference/upstream-ncm'),
                           '--source', str(SOURCE), '--output', str(partial), '--cfg-scale', '0',
                           '--prefix-json', str(input_path), '--decisions-only', '--stochastic', '--seed', str(seed)]
                run(command, log, env)
                partial.rename(result)
            else:
                command = [str(BINARY), 'sample-xsum', str(ARTIFACT), 'cuda-cudnn', str(input_path), '0', str(seed)]
                rust_env = env.copy(); rust_env['LD_LIBRARY_PATH'] = str(CUDNN)
                run(command, partial, rust_env, log)
                os.link(partial, result); partial.unlink()
        validate_batch(out, args.runtime, batch, args.seed * 1000 + batch, prefixes)
        print(json.dumps({'runtime': args.runtime, 'set': args.set, 'seed': args.seed,
                          'batch': batch, 'rows': 32}), flush=True)
    if (digest_file(DATA) != DATA_SHA or sources() != source_hashes
            or linked_libraries(BINARY, CUDNN) != libraries):
        raise ValueError('capture inputs changed')
    for name, (size, sha) in INPUTS.items():
        verify_input(SOURCE / name, size, sha)


if __name__ == '__main__':
    main()
