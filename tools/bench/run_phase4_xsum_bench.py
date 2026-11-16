"""Collect alternating, synchronized Rust/Python XSum generation timings on CUDA."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCHES = (1, 4)
REFERENCE = ROOT / '.reference/venv/bin/python'
BENCH = ROOT / 'target/release/examples/xsum_perf_screen'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-phase4-v2'
SOURCE = ROOT / 'artifacts/upstream/xsum-phase4'
INPUTS = ROOT / 'reports/phase4-xsum-real-inputs-v1/prefixes.json'
SOURCES = ['Cargo.toml', 'Cargo.lock', 'crates/ncm-core/Cargo.toml',
           'crates/ncm-model/Cargo.toml', 'crates/ncm-io/Cargo.toml',
           'crates/ncm-backend-cuda/Cargo.toml', 'crates/ncm-backend-cuda/build.rs',
           'crates/ncm-backend-cuda/examples/xsum_perf_screen.rs',
           'tools/reference/xsum_perf_screen.py', 'tools/reference/xsum_artifact.py',
           'tools/reference/tinystories_artifact.py', 'tools/bench/run_phase4_xsum_bench.py']
SOURCES += [str(p.relative_to(ROOT)) for root in ('crates/ncm-core/src', 'crates/ncm-model/src',
            'crates/ncm-io/src', 'crates/ncm-backend-cuda/src') for p in (ROOT/root).rglob('*') if p.is_file()]



def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def monitor(stop, path):
    queries = [
        ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv,noheader,nounits'],
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'],
    ]
    with path.open('x') as out:
        while not stop.is_set():
            sample = {'monotonic': time.monotonic(), 'queries': []}
            for command in queries:
                try:
                    r = subprocess.run(command, text=True, capture_output=True, timeout=5, check=False)
                    sample['queries'].append({'status': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
                except (OSError, subprocess.TimeoutExpired) as e:
                    sample['queries'].append({'error': str(e)})
            sample['parents'] = {}
            for row in sample['queries'][1].get('stdout', '').splitlines():
                try:
                    pid = int(row.split(',', 1)[0].strip())
                    status = Path(f'/proc/{pid}/status').read_text()
                    parent = next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:'))
                    sample['parents'][str(pid)] = int(parent)
                except (OSError, ValueError, StopIteration):
                    continue
            out.write(json.dumps(sample) + '\n'); out.flush()
            stop.wait(1)

def terminate_group(process):
    if process.poll() is not None: return
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3, choices=(1, 3))
    parser.add_argument('--cudnn-dir', type=Path, help='native cuDNN libraries for the explicitly built strict candidate')
    args = parser.parse_args()
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NCM_PHASE3_LT_PROFILE',
                 'NVIDIA_TF32_OVERRIDE', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
    if any(os.environ.get(name) for name in forbidden):
        raise SystemExit('profiling, injection or device-override environment is not permitted')
    if os.environ.get('LD_LIBRARY_PATH'):
        raise SystemExit('ambient LD_LIBRARY_PATH is not permitted')
    cudnn_dir = args.cudnn_dir.resolve() if args.cudnn_dir else None
    if cudnn_dir and not (cudnn_dir / 'libcudnn.so.9').is_file():
        raise SystemExit('strict candidate needs a native cuDNN 9 library directory')
    cudnn_files = sorted(cudnn_dir.glob('libcudnn*.so.9')) if cudnn_dir else []
    out = args.output.resolve()
    if out.exists(): raise SystemExit('output exists; choose a new run identity')
    if not args.cudnn_dir: raise SystemExit('strict cuDNN directory required')
    if digest(INPUTS) != 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf': raise SystemExit('input prefixes differ')
    if not BENCH.is_file() or not Path('/usr/bin/time').is_file(): raise SystemExit('benchmark or /usr/bin/time missing')
    if digest(ARTIFACT / 'manifest.json') != '5df37749e64418c586472a4678600603992357b9bc5e3611da307918e13b237e':
        raise SystemExit('XSum artifact manifest differs from the pinned export')
    out.mkdir(parents=True)
    paths = [ROOT / name for name in SOURCES] + [BENCH, ARTIFACT / 'manifest.json', INPUTS] + [SOURCE / 'config.yaml']
    manifest = {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'batches': BATCHES, 'warmup_per_process': 5, 'trials_per_process': 30, 'processes_per_runtime_batch': args.repetitions,
                'scope': 'eager generate and synchronized code/ID host readback; excludes load',
                'sampling': 'temperature 1, top-k 1, top-p 0, CFG 0; native rejects exact ties',
                'python_cleanup': 'previous GPU outputs released outside generation timing',
                'rust_backend': 'strict cuDNN' if cudnn_dir else 'default CUDA',
                'input_sha256': digest(INPUTS), 'cudnn_sha256': {p.name: digest(p) for p in cudnn_files},
                'order': 'alternating runtime per batch/repetition', 'telemetry': 'nvidia-smi approximately once per second',
                'clock_policy': 'unchanged', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'environment': {name: subprocess.check_output(command, cwd=ROOT, text=True).strip() for name, command in {
                    'rustc': ['rustc', '--version'], 'nvcc': ['nvcc', '--version'],
                    'kernel': ['uname', '-a'], 'python': [str(REFERENCE), '--version'],
                    'gpu': ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']}.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    stop = threading.Event(); thread = threading.Thread(target=monitor, args=(stop, out/'activity.jsonl'), daemon=True); thread.start()
    process = None
    expected_outputs = {}
    try:
        for j, batch in enumerate(BATCHES):
            for rep in range(args.repetitions):
                for runtime in (('rust', 'python') if (j+rep)%2 == 0 else ('python', 'rust')):
                    stem=f'{runtime}-b{batch}-r{rep}'
                    if runtime == 'rust':
                        command=[str(BENCH), str(ARTIFACT), str(INPUTS), str(batch), 'device', '30']
                    else:
                        command=[str(REFERENCE), str(ROOT/'tools/reference/xsum_perf_screen.py'),
                                 '--upstream',str(ROOT/'.reference/upstream-ncm'), '--source',str(SOURCE), '--inputs',str(INPUTS), '--batch',str(batch), '--no-stage-sync', '--trials','30']
                    env=os.environ.copy(); env['OMP_NUM_THREADS']='1'; env['TOKENIZERS_PARALLELISM']='false'
                    if runtime == 'rust' and cudnn_dir: env['LD_LIBRARY_PATH'] = str(cudnn_dir)
                    with (out/(stem+'.stdout')).open('xb') as stdout, (out/(stem+'.stderr')).open('xb') as stderr:
                        process=subprocess.Popen(['/usr/bin/time','-v','-o',str(out/(stem+'.time')),*command], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                        with (out/'process-events.jsonl').open('a') as events:
                            events.write(json.dumps({'event':'start','name':stem,'pid':process.pid,'monotonic':time.monotonic()})+'\n')
                        try: code=process.wait(timeout=600)
                        except subprocess.TimeoutExpired:
                            terminate_group(process); raise RuntimeError(f'{stem} timed out')
                    with (out/'process-events.jsonl').open('a') as events:
                        events.write(json.dumps({'event':'end','name':stem,'monotonic':time.monotonic(),'returncode':code})+'\n')
                    process=None
                    if code: raise RuntimeError(f'{stem} exited {code}')
                    lines=(out/(stem+'.stdout')).read_text().splitlines()
                    if not lines: raise ValueError(f'{stem} produced no output')
                    result=json.loads(lines[-1])
                    (out/(stem+'.json')).write_text(json.dumps(result, sort_keys=True)+'\n')
                    if result['batch'] != batch or result['warmups'] != 5 or len(result['trials_seconds']) != 30:
                        raise ValueError(f'{stem} has incomplete trials')
                    if runtime == 'rust' and result.get('backend') != ('strict-cudnn' if cudnn_dir else 'cuda'):
                        raise ValueError(f'{stem} backend does not match the requested candidate')
                    codes = result['codes'] if runtime == 'rust' else [x for row in result['codes'] for x in row]
                    ids = result['ids'] if runtime == 'rust' else [x for row in result['ids'] for x in row]
                    if len(codes) != batch * 200 or len(ids) != batch * 48:
                        raise ValueError(f'{stem} output shape differs')
                    signature = (codes, ids)
                    if batch in expected_outputs and expected_outputs[batch] != signature:
                        raise ValueError(f'{stem} output differs from the paired runtime')
                    expected_outputs[batch] = signature
                    print(stem, 'complete', flush=True)
    finally:
        if process is not None: terminate_group(process)
        stop.set(); thread.join(timeout=10)
    for path in paths:
        if digest(path) != manifest['source_sha256'][str(path.relative_to(ROOT))]:
            raise RuntimeError(f'source changed during benchmark: {path}')
    for path in cudnn_files:
        if digest(path) != manifest['cudnn_sha256'][path.name]:
            raise RuntimeError(f'cuDNN library changed during benchmark: {path}')
    print('complete', flush=True)

if __name__ == '__main__': main()
