"""Alternating fresh-process LM1B (small, medium or large) multi-batch development timing screen."""
import argparse
import json
import math
import os
import re
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
SIZES = ('small', 'medium', 'large')
CAPTURE = ROOT / 'reports/phase4-lm1b-small-device-dev-seed-11-v1.json'
REFERENCE = ROOT / 'reports/phase4-lm1b-small-dev-seed-11-v1.json'



def summarize(rows, trials=8, batches=(1, 4, 16)):
    summaries = {}
    for batch in batches:
        summaries[str(batch)] = {}
        for runtime in ('python', 'rust'):
            selected = [row for row in rows if row['runtime'] == runtime and row['batch'] == batch]
            if len(selected) != 3 or {row['rep'] for row in selected} != {0, 1, 2}:
                raise ValueError(f'incomplete repetitions for {runtime} batch {batch}')
            if any(row['signatures'] != selected[0]['signatures'] for row in selected):
                raise ValueError(f'output signatures differ for {runtime} batch {batch}')
            values = sorted(x for row in selected for x in row['seconds'])
            if (any(len(row['seconds']) != trials for row in selected) or
                    any(not math.isfinite(x) or x <= 0 for x in values)):
                raise ValueError(f'invalid timings for {runtime} batch {batch}')
            summaries[str(batch)][runtime] = {'median_seconds': statistics.median(values), 'p95_seconds': values[math.ceil(len(values) * .95) - 1],
                                  'samples': len(values), 'samples_per_second_at_median': batch / statistics.median(values)}
    if len(rows) != 6 * len(batches):
        raise ValueError('unexpected extra results')
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference-rope-diagnostic', type=Path)
    p.add_argument('--repeated', action='store_true')
    p.add_argument('--size', choices=SIZES, default='small')
    p.add_argument('--batch', type=int, choices=(1, 4, 8, 16),
                   help='measure only this batch size (default: 1, 4 and 16)')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--rust-cudnn-default-math', action='store_true')
    modes.add_argument('--rust-cudnn-graph', action='store_true')
    modes.add_argument('--rust-cudnn-graph-efficient', action='store_true')
    args = p.parse_args()
    batches = (args.batch,) if args.batch is not None else (1, 4, 16)
    artifact = ROOT / f'artifacts/canonical/lm1b-{args.size}-inference-v1'
    if (args.rust_cudnn_default_math or args.rust_cudnn_graph or args.rust_cudnn_graph_efficient) and args.reference_rope_diagnostic is None:
        p.error('experimental cuDNN modes require --reference-rope-diagnostic')
    rust_mode, convolution_math = None, 'strict FMA'
    if args.rust_cudnn_graph_efficient:
        rust_mode, convolution_math = 'cudnn-graph-efficient', 'cuDNN graph and efficient attention'
    elif args.rust_cudnn_graph:
        rust_mode, convolution_math = 'cudnn-graph', 'cuDNN graph'
    elif args.rust_cudnn_default_math:
        rust_mode, convolution_math = 'cudnn-default-math', 'cuDNN default math'
    warmups, trials = (5, 30) if args.repeated else (2, 8)
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output directory under reports required')
    if any(os.environ.get(key) for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                                            'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('unexpected environment override')
    source = [Path(__file__).resolve(), ROOT / 'tools/reference/lm1b_small_capacity_probe.py',
              ROOT / 'tools/bench/run_phase4_xsum_bench.py', RUST, artifact / 'manifest.json',
              ROOT / 'Cargo.lock', ROOT / 'crates/ncm-backend-cuda/examples/lm1b_small_capacity_probe.rs']
    source.append(ROOT / 'crates/ncm-backend-cuda/examples/support/cuda_pool.rs')
    for directory in ['crates/ncm-core/src', 'crates/ncm-model/src', 'crates/ncm-backend-cuda/src']:
        source.extend(sorted(p for p in (ROOT / directory).iterdir() if p.suffix in ('.rs', '.cu')))
    source.extend(sorted(CUDNN.glob('libcudnn*.so.9')))
    if args.reference_rope_diagnostic:
        if digest(args.reference_rope_diagnostic) != '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc':
            raise ValueError('unexpected diagnostic rotary tables')
        source.append(args.reference_rope_diagnostic.resolve())
    before = {str(path): digest(path) for path in source}
    out.mkdir(parents=True)
    manifest = {'scope': f'non-held-out LM1B-{args.size} multi-batch development performance screen',
                'size': args.size,
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'sources_sha256': before, 'order': 'alternating runtime per fresh process',
                'warmups': warmups, 'trials': trials, 'repetitions': 3, 'batches': batches, 'rust_penalty': 'device',
                'rust_diagnostic': 'reference tables and 1 MiB QKV preference' if args.reference_rope_diagnostic else 'default',
                'rust_convolution_math': convolution_math,
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
        for batch in batches:
            for rep in range(3):
                for runtime in (('python', 'rust') if rep % 2 == 0 else ('rust', 'python')):
                    stem = f'{runtime}-b{batch}-r{rep}'
                    if runtime == 'rust':
                        command = [str(RUST), str(artifact), str(batch)]
                    else:
                        command = [str(PYTHON), str(ROOT / 'tools/reference/lm1b_small_capacity_probe.py'),
                                   '--upstream', str(ROOT / '.reference/upstream-ncm'), '--source', str(ROOT / 'artifacts/upstream/lm1b-phase4'),
                                   '--size', args.size, '--seed', '11', '--batch', str(batch), '--output', str(out / (stem + '.json'))]
                    if runtime == 'rust' and rust_mode:
                        command.append(rust_mode)
                    if args.repeated:
                        command.append('repeated' if runtime == 'rust' else '--repeated')
                    env = os.environ.copy()
                    env.pop('NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC', None)
                    env.update({'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'LD_LIBRARY_PATH': str(CUDNN)})
                    if runtime == 'rust' and args.reference_rope_diagnostic:
                        env['NCM_LM1B_REFERENCE_ROPE_DIAGNOSTIC'] = str(args.reference_rope_diagnostic.resolve())
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
                        expected_diagnostic = 'reference tables and 1 MiB QKV preference' if args.reference_rope_diagnostic else 'default'
                        if result.get('diagnostic') != expected_diagnostic:
                            raise ValueError('unexpected Rust diagnostic selection')
                        if result.get('convolution_math') != convolution_math:
                            raise ValueError('unexpected Rust convolution math')
                    else:
                        result = json.loads((out / (stem + '.json')).read_bytes())
                    values = result['seconds']
                    if (result['runtime'] != runtime or result['batch'] != batch or result['trials'] != trials or
                            result['warmups'] != warmups or len(values) != trials or
                            any(not math.isfinite(x) or x <= 0 for x in values) or
                            len(result['signatures']) != trials or any(len(x) != 64 for x in result['signatures'])):
                        raise ValueError('invalid probe result')
                    rss = re.fullmatch(r'VmHWM:\s+([1-9]\d*) kB', result['vmhwm'])
                    load_seconds = result['load_seconds']
                    if rss is None or not math.isfinite(load_seconds) or load_seconds <= 0:
                        raise ValueError('invalid process RSS or load time')
                    if runtime == 'rust':
                        memory = result['cuda_default_pool']
                        fields = ('used_current', 'used_peak', 'reserved_current', 'reserved_peak')
                    else:
                        fields = ('torch_active_bytes', 'torch_peak_allocated_bytes',
                                  'torch_reserved_bytes', 'torch_peak_reserved_bytes')
                        memory = {field: result[field] for field in fields}
                    if set(memory) != set(fields) or any(type(v) is not int or v < 0 for v in memory.values()):
                        raise ValueError('invalid native allocator counters')
                    rows.append({'runtime': runtime, 'batch': batch, 'rep': rep, 'signatures': result['signatures'],
                                 'pid': pid, 'start_monotonic': started, 'end_monotonic': time.monotonic(),
                                 'seconds': values, 'vmhwm': result['vmhwm'], 'peak_rss_kib': int(rss[1]),
                                 'load_seconds': load_seconds, 'native_allocator_bytes': memory})
    finally:
        if process is not None:
            terminate_group(process)
        stop.set()
        thread.join()
    if before != {str(path): digest(path) for path in source}:
        raise ValueError('source or capture changed')
    summaries = summarize(rows, trials, batches)
    result = {'scope': manifest['scope'], 'rows': rows, 'summary': summaries,
              'limits': 'separate random streams; desktop GPU activity; device process telemetry is sampled and allocator counters have runtime-specific scopes'}
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(summaries, sort_keys=True))


if __name__ == '__main__':
    main()
