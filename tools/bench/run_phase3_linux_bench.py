"""Collect alternating, synchronized Rust/Python Sudoku generation timings on CUDA."""
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
BATCHES = (1, 4, 32, 128)
REFERENCE = ROOT / '.reference/venv/bin/python'
BENCH = ROOT / 'target/release/examples/generator_bench'
ARTIFACT = ROOT / 'artifacts/canonical/sudoku-easy-qualified'
DATA = ROOT / 'artifacts/canonical/sudoku-data-phase3'
SOURCES = ['Cargo.toml', 'Cargo.lock', 'crates/ncm-backend-cuda/Cargo.toml',
           'tools/reference/sudoku_artifact.py', 'crates/ncm-model/src/generator.rs', 'crates/ncm-model/src/codec.rs',
           'crates/ncm-model/src/blocks.rs', 'crates/ncm-model/src/lib.rs',
           'crates/ncm-core/src/lib.rs', 'crates/ncm-core/src/config.rs',
           'crates/ncm-backend-cuda/src/codec.rs', 'crates/ncm-backend-cuda/src/ops.rs',
           'crates/ncm-backend-cuda/src/runtime.rs', 'crates/ncm-backend-cuda/src/resample.rs',
           'crates/ncm-backend-cuda/build.rs', 'crates/ncm-backend-cuda/src/generator_layout.rs',
           'crates/ncm-backend-cuda/src/generator_layout.cu',
           'crates/ncm-backend-cuda/src/fused_projection.rs',
           'crates/ncm-backend-cuda/src/fused_projection/bounded.rs', 'crates/ncm-io/src/lib.rs',
           'crates/ncm-backend-cuda/examples/generator_bench.rs',
           'tools/reference/phase3_generator_reference.py', 'tools/bench/run_phase3_linux_bench.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

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
    args = parser.parse_args()
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NCM_PHASE3_LT_PROFILE')
    if any(os.environ.get(name) for name in forbidden):
        raise SystemExit('profiling, injection or device-override environment is not permitted')
    out = args.output.resolve()
    if out.exists(): raise SystemExit('output exists; choose a new run identity')
    if not BENCH.is_file() or not Path('/usr/bin/time').is_file(): raise SystemExit('benchmark or /usr/bin/time missing')
    out.mkdir(parents=True)
    paths = [ROOT / name for name in SOURCES] + [BENCH, ARTIFACT / 'manifest.json'] + [DATA / f'easy-{n}-inputs.json' for n in BATCHES]
    manifest = {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'batches': BATCHES, 'warmup_per_process': 5, 'trials_per_process': 30, 'processes_per_runtime_batch': args.repetitions,
                'scope': 'eager generate and synchronized code/ID host readback; excludes load',
                'order': 'alternating runtime per batch/repetition', 'telemetry': 'nvidia-smi approximately once per second',
                'clock_policy': 'unchanged', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'environment': {name: subprocess.check_output(command, cwd=ROOT, text=True).strip() for name, command in {
                    'rustc': ['rustc', '--version'], 'nvcc': ['nvcc', '--version'],
                    'kernel': ['uname', '-a'], 'python': [str(REFERENCE), '--version'],
                    'gpu': ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']}.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    stop = threading.Event(); thread = threading.Thread(target=monitor, args=(stop, out/'activity.jsonl'), daemon=True); thread.start()
    process = None
    try:
        for j, batch in enumerate(BATCHES):
            for rep in range(args.repetitions):
                for runtime in (('rust', 'python') if (j+rep)%2 == 0 else ('python', 'rust')):
                    stem=f'{runtime}-b{batch}-r{rep}'
                    input_path=DATA/f'easy-{batch}-inputs.json'
                    if runtime == 'rust':
                        command=[str(BENCH), str(ARTIFACT), str(input_path), str(batch), 'cuda']
                    else:
                        command=[str(REFERENCE), str(ROOT/'tools/reference/phase3_generator_reference.py'),
                                 '--difficulty','easy','--upstream',str(ROOT/'.reference/upstream-ncm'),
                                 '--source',str(ROOT/'artifacts/upstream/sudoku/easy'), '--inputs',str(input_path),
                                 '--output',str(out/(stem+'.json')), '--batch-size',str(batch), '--bench']
                    env=os.environ.copy(); env['OMP_NUM_THREADS']='1'; env['TOKENIZERS_PARALLELISM']='false'
                    with (out/(stem+'.stdout')).open('xb') as stdout, (out/(stem+'.stderr')).open('xb') as stderr:
                        process=subprocess.Popen(['/usr/bin/time','-v','-o',str(out/(stem+'.time')),*command], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                        with (out/'process-events.jsonl').open('a') as events:
                            events.write(json.dumps({'event':'start','name':stem,'pid':process.pid,'monotonic':time.monotonic()})+'\n')
                        try: code=process.wait(timeout=180)
                        except subprocess.TimeoutExpired:
                            terminate_group(process); raise RuntimeError(f'{stem} timed out')
                    with (out/'process-events.jsonl').open('a') as events:
                        events.write(json.dumps({'event':'end','name':stem,'monotonic':time.monotonic(),'returncode':code})+'\n')
                    process=None
                    if code: raise RuntimeError(f'{stem} exited {code}')
                    if runtime == 'rust': (out/(stem+'.json')).write_bytes((out/(stem+'.stdout')).read_bytes())
                    result=json.loads((out/(stem+'.json')).read_text())
                    if result['batch'] != batch or result['warmups'] != 5 or len(result['trials']) != 30:
                        raise ValueError(f'{stem} has incomplete trials')
                    print(stem, 'complete', flush=True)
    finally:
        if process is not None: terminate_group(process)
        stop.set(); thread.join(timeout=10)
    print('complete', flush=True)

if __name__ == '__main__': main()
