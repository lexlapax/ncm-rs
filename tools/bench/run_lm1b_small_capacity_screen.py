"""Alternating fresh-process LM1B multi-batch development timing screen."""
import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

from run_phase4_xsum_bench import digest, monitor, terminate_group

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.reference/venv/bin/python'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
RUST = ROOT / 'target/release/examples/lm1b_small_capacity_probe'
ARTIFACT = ROOT / 'artifacts/canonical/lm1b-small-inference-v1'
CAPTURE = ROOT / 'reports/phase4-lm1b-small-device-dev-seed-11-v1.json'
REFERENCE = ROOT / 'reports/phase4-lm1b-small-dev-seed-11-v1.json'



def summarize(rows):
    summaries = {}
    for batch in (1, 4, 16):
        summaries[str(batch)] = {}
        for runtime in ('python', 'rust'):
            selected = [row for row in rows if row['runtime'] == runtime and row['batch'] == batch]
            if len(selected) != 3 or {row['rep'] for row in selected} != {0, 1, 2}:
                raise ValueError(f'incomplete repetitions for {runtime} batch {batch}')
            if any(row['signatures'] != selected[0]['signatures'] for row in selected):
                raise ValueError(f'output signatures differ for {runtime} batch {batch}')
            values = sorted(x for row in selected for x in row['seconds'])
            if len(values) != 24 or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError(f'invalid timings for {runtime} batch {batch}')
            summaries[str(batch)][runtime] = {'median_seconds': statistics.median(values), 'p95_seconds': values[math.ceil(len(values) * .95) - 1],
                                  'samples': len(values), 'samples_per_second_at_median': batch / statistics.median(values)}
    if len(rows) != 18:
        raise ValueError('unexpected extra results')
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output directory under reports required')
    if any(os.environ.get(key) for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                                            'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('unexpected environment override')
    source = [Path(__file__).resolve(), ROOT / 'tools/reference/lm1b_small_capacity_probe.py',
              ROOT / 'tools/bench/run_phase4_xsum_bench.py', RUST, ARTIFACT / 'manifest.json',
              ROOT / 'Cargo.lock', ROOT / 'crates/ncm-backend-cuda/examples/lm1b_small_capacity_probe.rs']
    for directory in ['crates/ncm-core/src', 'crates/ncm-model/src', 'crates/ncm-backend-cuda/src']:
        source.extend(sorted(p for p in (ROOT / directory).iterdir() if p.suffix in ('.rs', '.cu')))
    source.extend(sorted(CUDNN.glob('libcudnn*.so.9')))
    before = {str(path): digest(path) for path in source}
    out.mkdir(parents=True)
    manifest = {'scope': 'non-held-out LM1B-small multi-batch development performance screen; not a gate',
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'sources_sha256': before, 'order': 'alternating runtime per fresh process',
                'warmups': 2, 'trials': 8, 'repetitions': 3, 'batches': [1, 4, 16], 'rust_penalty': 'device',
                'telemetry': 'nvidia-smi approximately once per second',
                'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl'), daemon=True)
    thread.start()
    process = None
    rows = []
    try:
        for batch in (1, 4, 16):
            for rep in range(3):
                for runtime in (('python', 'rust') if rep % 2 == 0 else ('rust', 'python')):
                    stem = f'{runtime}-b{batch}-r{rep}'
                    if runtime == 'rust':
                        command = [str(RUST), str(ARTIFACT), str(batch)]
                    else:
                        command = [str(PYTHON), str(ROOT / 'tools/reference/lm1b_small_capacity_probe.py'),
                                   '--upstream', str(ROOT / '.reference/upstream-ncm'), '--source', str(ROOT / 'artifacts/upstream/lm1b-phase4'),
                                   '--size', 'small', '--seed', '11', '--batch', str(batch), '--output', str(out / (stem + '.json'))]
                    env = os.environ.copy()
                    env.update({'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'LD_LIBRARY_PATH': str(CUDNN)})
                    with (out / (stem + '.log')).open('x') as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        pid, started = process.pid, time.monotonic()
                        try:
                            code = process.wait(timeout=300)
                        except subprocess.TimeoutExpired:
                            terminate_group(process)
                            raise
                    if code:
                        raise RuntimeError(f'{stem} exited {code}; see retained log')
                    if runtime == 'rust':
                        result = json.loads((out / (stem + '.log')).read_text().splitlines()[-1])
                    else:
                        result = json.loads((out / (stem + '.json')).read_bytes())
                    values = result['seconds']
                    if (result['runtime'] != runtime or result['batch'] != batch or result['trials'] != 8 or
                            result['warmups'] != 2 or len(values) != 8 or
                            any(not math.isfinite(x) or x <= 0 for x in values) or
                            len(result['signatures']) != 8 or any(len(x) != 64 for x in result['signatures'])):
                        raise ValueError('invalid probe result')
                    rows.append({'runtime': runtime, 'batch': batch, 'rep': rep, 'signatures': result['signatures'], 'pid': pid, 'start_monotonic': started,
                                 'end_monotonic': time.monotonic(), 'seconds': values, 'vmhwm': result['vmhwm']})
    finally:
        if process is not None:
            terminate_group(process)
        stop.set()
        thread.join()
    if before != {str(path): digest(path) for path in source}:
        raise ValueError('source or capture changed')
    summaries = summarize(rows)
    result = {'scope': manifest['scope'], 'rows': rows, 'summary': summaries,
              'limits': 'separate random streams; desktop GPU activity; memory telemetry is sampled, not an allocator peak'}
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(summaries, sort_keys=True))


if __name__ == '__main__':
    main()
