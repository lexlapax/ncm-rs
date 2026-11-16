"""Experimental resumable Rust XSum batch driver; requires review and GPU validation."""
import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, 'tools/reference')
from xsum_quality_generate import DATA_SHA, ROOT, canonical, shard_batches
from xsum_artifact import digest_file
from xsum_linked_libraries import linked_libraries

ARTIFACT = ROOT / 'artifacts/canonical/xsum-phase4-v2'
ARTIFACT_SHA = '5df37749e64418c586472a4678600603992357b9bc5e3611da307918e13b237e'
ROPE_ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
ROPE_ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
BINARY = ROOT / 'target/release/examples/xsum_quality_batches'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'


def input_hashes(artifact=ARTIFACT):
    binaries = [BINARY, artifact / 'manifest.json'] + sorted(CUDNN.glob('libcudnn*.so.9'))
    if not BINARY.is_file() or not (CUDNN / 'libcudnn.so.9').is_file():
        raise ValueError('missing CUDA build or libraries')
    source_files = [Path(__file__).resolve(), ROOT / 'Cargo.toml', ROOT / 'Cargo.lock',
                    ROOT / 'tools/reference/xsum_quality_generate.py', ROOT / 'tools/reference/xsum_artifact.py',
                    ROOT / 'tools/reference/tinystories_artifact.py', ROOT / 'tools/reference/xsum_rope_reference.py',
                    ROOT / 'tools/reference/xsum_linked_libraries.py',
                    ROOT / 'crates/ncm-backend-cuda/Cargo.toml', ROOT / 'crates/ncm-backend-cuda/build.rs',
                    ROOT / 'crates/ncm-core/Cargo.toml', ROOT / 'crates/ncm-model/Cargo.toml',
                    ROOT / 'crates/ncm-io/Cargo.toml', ROOT / 'crates/ncm-backend-cuda/examples/xsum_quality_batches.rs']
    source_files += [p for directory in ('crates/ncm-core/src', 'crates/ncm-model/src', 'crates/ncm-io/src', 'crates/ncm-backend-cuda/src') for p in (ROOT / directory).rglob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)): digest_file(p) for p in binaries + source_files}


def read_batch(path, indices, ordinal):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('invalid batch file')
    envelope = json.loads(path.read_bytes())
    if set(envelope) != {'record', 'sha256'} or hashlib.sha256(canonical(envelope['record'])).hexdigest() != envelope['sha256']:
        raise ValueError('batch digest differs')
    record = envelope['record']
    if set(record) != {'ordinal', 'indices', 'codes', 'raw_ids'} or record['ordinal'] != ordinal or record['indices'] != indices:
        raise ValueError('batch identity differs')
    for key, width, bound in [('codes', 200, 4096), ('raw_ids', 48, 50304)]:
        rows = record[key]
        if type(rows) is not list or len(rows) != len(indices) or any(
            type(row) is not list or len(row) != width or any(type(x) is not int or x < 0 or x >= bound for x in row) for row in rows
        ):
            raise ValueError('batch dimensions or values differ')
    return record


def bounded_line(pipe, timeout=180):
    data = bytearray()
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(pipe, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not sel.select(remaining):
                raise TimeoutError('Rust quality batch timed out')
            chunk = os.read(pipe.fileno(), 65536)
            if not chunk:
                raise RuntimeError('Rust quality process closed its output')
            data.extend(chunk)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError('Rust quality output too large')
            if b'\n' in data:
                if not data.endswith(b'\n') or data.count(b'\n') != 1:
                    raise ValueError('unexpected extra output')
                return json.loads(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', type=int, choices=(0, 1), required=True)
    ap.add_argument('--stop', type=int, default=11334)
    ap.add_argument('--profile', choices=('v1', 'v2'), default='v1')
    args = ap.parse_args()
    artifact, artifact_sha = ((ARTIFACT, ARTIFACT_SHA) if args.profile == 'v1'
                              else (ROPE_ARTIFACT, ROPE_ARTIFACT_SHA))
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')
    if any(os.environ.get(name) for name in forbidden): raise ValueError('unsupported environment override')
    if args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be under reports')
    if args.data.is_symlink() or digest_file(args.data) != DATA_SHA or digest_file(artifact / 'manifest.json') != artifact_sha:
        raise ValueError('dataset or artifact differs')
    data = json.loads(args.data.read_bytes())
    if data.get('profile') != 'xsum' or len(data.get('inputs', [])) != 11334: raise ValueError('bad data')
    hashes = input_hashes(artifact)
    native_libraries = linked_libraries(BINARY, CUDNN)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1: raise ValueError('exactly one GPU required')
    manifest = {'schema_version': 1, 'origin': 'rust-strict-cudnn', 'shard': args.shard, 'shards': 2,
                'data_sha256': DATA_SHA, 'artifact_sha256': artifact_sha, 'artifact_profile': args.profile,
                'binary_and_source_sha256': hashes, 'native_libraries': native_libraries,
                'gpu_identity_and_driver': gpu[0], 'batch_size': 32, 'temperature': 1.0,
                'top_k': 1, 'top_p': 0.0, 'cfg': 0.0}
    out = args.output.resolve()
    if out.exists():
        meta = out / 'manifest.json'
        if not out.is_dir() or meta.is_symlink() or not meta.is_file() or json.loads(meta.read_bytes()) != manifest:
            raise ValueError('manifest differs')
    else:
        out.mkdir(parents=True)
        with (out / 'manifest.json').open('x') as f: json.dump(manifest, f, sort_keys=True)
    batches, selected = shard_batches(args.shard, args.stop)
    existing = sorted(out.glob('batch-*.json'))
    if [p.name for p in existing] != [f'batch-{i:05}.json' for i in range(len(existing))] or len(existing) > len(selected):
        raise ValueError('batch sequence or requested stop differs')
    for i, path in enumerate(existing): read_batch(path, batches[i], i)
    if len(existing) == len(selected): return
    env = os.environ.copy(); env['LD_LIBRARY_PATH'] = str(CUDNN); env['OMP_NUM_THREADS'] = '1'
    with (out / 'worker.stderr').open('ab') as error:
        process = subprocess.Popen([str(BINARY), str(artifact)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=error, env=env, start_new_session=True)
        try:
            for ordinal, indices in selected[len(existing):]:
                prefixes = [(data['inputs'][i][:512] + [50257] * 512)[:512] for i in indices]
                process.stdin.write(canonical(prefixes) + b'\n'); process.stdin.flush()
                output = bounded_line(process.stdout)
                record = {'ordinal': ordinal, 'indices': indices, 'codes': output['codes'], 'raw_ids': output['raw_ids']}
                envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
                tmp = out / f'batch-{ordinal:05}.{os.getpid()}.part'
                with tmp.open('xb') as f: f.write(canonical(envelope) + b'\n'); f.flush(); os.fsync(f.fileno())
                read_batch(tmp, indices, ordinal)
                os.link(tmp, out / f'batch-{ordinal:05}.json'); tmp.unlink()
                fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
                print(json.dumps({'shard': args.shard, 'ordinal': ordinal, 'rows': len(indices)}), flush=True)
            process.stdin.close()
            if process.wait(timeout=15): raise RuntimeError('Rust quality process failed')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
    if (digest_file(args.data) != DATA_SHA or input_hashes(artifact) != hashes
            or linked_libraries(BINARY, CUDNN) != native_libraries):
        raise ValueError('quality source changed during execution')


if __name__ == '__main__': main()
