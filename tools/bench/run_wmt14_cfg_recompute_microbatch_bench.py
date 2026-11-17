"""Run alternating fresh-process WMT14 CFG recomputation microbatch CUDA timing comparisons."""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from run_phase4_xsum_bench import digest, monitor, terminate_group

PREFIX = ROOT / 'reports/phase4-wmt14-tokenizer-reference-v1.json'
ARTIFACT = ROOT / 'artifacts/canonical/wmt14-inference-v1-final'
BENCH = ROOT / 'target/release/examples/wmt14_cfg_recompute_microbatch_perf'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
PYTHON = ROOT / '.reference/venv/bin/python'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, choices=(1, 3), default=3)
    p.add_argument('--batches', type=int, nargs='+', choices=(1, 4, 16), default=[1, 4, 16])
    p.add_argument('--modes', type=int, nargs='+', choices=(1,), default=[1])
    args = p.parse_args()
    if 1 in args.modes and 32 in args.batches:
        raise ValueError('CFG doubles the internal batch beyond the candidate limit')
    if len(set(args.batches)) != len(args.batches) or len(set(args.modes)) != len(args.modes):
        raise ValueError('duplicate workload')
    forbidden = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING',
                 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
    if any(os.environ.get(key) for key in forbidden):
        raise ValueError('unexpected environment override')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be a new directory under reports')
    if digest(PREFIX) != '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f':
        raise ValueError('prefixes changed')
    if digest(ARTIFACT / 'manifest.json') != '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81':
        raise ValueError('artifact changed')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError('tracked source must be clean')
    watched = [BENCH, PREFIX, ARTIFACT / 'manifest.json', Path(__file__), ROOT / 'tools/reference/wmt14_perf_screen.py',
               ROOT / 'tools/bench/run_phase4_xsum_bench.py', *sorted(CUDNN.glob('libcudnn*.so.9'))]
    before = {str(path): digest(path) for path in watched}
    out.mkdir(parents=True)
    manifest = {'head': head, 'sha256': before, 'batches': args.batches, 'modes': args.modes,
                'repetitions': args.repetitions, 'warmups': 5, 'trials': 30,
                'scope': 'prefix upload, generation, decode and synchronized code/ID readback; model load excluded',
                'order': 'alternating runtime per workload and repetition', 'telemetry': 'nvidia-smi approximately once per second',
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
                'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
                'python': subprocess.check_output([str(PYTHON), '--version'], text=True).strip()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl'), daemon=True)
    thread.start()
    process = None
    signatures, measurements = {}, {}
    try:
        for mode in args.modes:
            for batch in args.batches:
                for rep in range(args.repetitions):
                    for runtime in (('rust', 'python') if (mode + batch + rep) % 2 == 0 else ('python', 'rust')):
                        stem = f'{runtime}-cfg{mode}-b{batch}-r{rep}'
                        if runtime == 'rust':
                            command = [str(BENCH), str(ARTIFACT), str(PREFIX), str(batch), str(mode)]
                        else:
                            command = [str(PYTHON), str(ROOT / 'tools/reference/wmt14_perf_screen.py'),
                                       '--upstream', str(ROOT / '.reference/upstream-ncm'), '--source', str(ROOT / 'artifacts/upstream/wmt14-phase4'),
                                       '--prefixes', str(PREFIX), '--output', str(out / (stem + '.json')), '--batch', str(batch), '--cfg', str(mode)]
                        env = os.environ.copy()
                        env.update(OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
                        if runtime == 'rust':
                            env['LD_LIBRARY_PATH'] = str(CUDNN)
                        with (out / (stem + '.stdout')).open('xb') as stdout, (out / (stem + '.stderr')).open('xb') as stderr:
                            process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(out / (stem + '.time')), *command], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                            with (out / 'process-events.jsonl').open('a') as events:
                                events.write(json.dumps({'event': 'start', 'name': stem, 'pid': process.pid, 'monotonic': time.monotonic()}) + '\n')
                            try:
                                code = process.wait(timeout=1200)
                            except subprocess.TimeoutExpired:
                                terminate_group(process)
                                raise RuntimeError(f'{stem} timed out')
                        with (out / 'process-events.jsonl').open('a') as events:
                            events.write(json.dumps({'event': 'end', 'name': stem, 'monotonic': time.monotonic(), 'returncode': code}) + '\n')
                        process = None
                        if code:
                            raise RuntimeError(f'{stem} exited {code}')
                        if runtime == 'rust':
                            result = json.loads((out / (stem + '.stdout')).read_text().splitlines()[-1])
                            (out / (stem + '.json')).write_text(json.dumps(result, sort_keys=True) + '\n')
                        else:
                            result = json.loads((out / (stem + '.json')).read_text())
                        times = result['trials_seconds']
                        if result['batch'] != batch or result['cfg'] != mode or result['warmups'] != 5 or len(times) != 30 or any(not math.isfinite(x) or x <= 0 for x in times):
                            raise ValueError(f'{stem} incomplete timing')
                        device_times = result['device_event_seconds']
                        if len(device_times) != 30 or any(not math.isfinite(x) or x <= 0 for x in device_times):
                            raise ValueError(f'{stem} incomplete device events')
                        outputs = result['outputs']
                        if len(outputs['codes']) != batch or len(outputs['ids']) != batch or any(len(row) != 313 for row in outputs['codes']) or any(len(row) != 64 for row in outputs['ids']):
                            raise ValueError(f'{stem} output shape')
                        key = (mode, batch)
                        if key in signatures and signatures[key] != outputs:
                            raise ValueError(f'{stem} outputs differ')
                        signatures[key] = outputs
                        measurements.setdefault(f'cfg{mode}-b{batch}-{runtime}', []).append({'median_seconds': statistics.median(times), 'trials_seconds': times, 'device_event_seconds': device_times, 'peak_rss_kib': result['peak_rss_kib'], 'cleanup_seconds': result.get('cleanup_seconds'), 'peak_allocated_bytes': result.get('peak_allocated_bytes'), 'peak_reserved_bytes': result.get('peak_reserved_bytes')})
                        print(stem, 'median', statistics.median(times), flush=True)
    finally:
        if process is not None:
            terminate_group(process)
        stop.set()
        thread.join()
    if before != {str(path): digest(path) for path in watched} or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise ValueError('source changed during run')
    (out / 'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')


if __name__ == '__main__':
    main()
