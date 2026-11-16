"""Compare baseline and candidate Metal generators in fresh alternating processes."""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import threading
from pathlib import Path

from run_phase3_mlx_bench import digest, monitor, run
from phase4_mlx_gpu_activity import monitor_gpu

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'artifacts/canonical/tinystories-phase4-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--candidate-source', type=Path, action='append', default=[],
                        help='isolated candidate source to fingerprint; may be repeated')
    parser.add_argument('--batches', type=int, nargs='+', choices=(32, 128), default=(32, 128),
                        help='batches to measure; default is both')
    parser.add_argument('--gpu-activity', action='store_true',
                        help='sample system-wide registry counters; diagnostic results may be perturbed')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.batches)) != len(args.batches):
        parser.error('batches must be unique')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise ValueError('instrumented environment is not permitted')
    old, new, out = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    if out.exists() or not old.is_file() or not new.is_file() or old == new:
        raise ValueError('need distinct binaries and a new output directory')
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
               ROOT / 'tools/bench/phase3_process.py', ROOT / 'tools/bench/phase4_mlx_gpu_activity.py',
               ROOT / 'Cargo.toml', ROOT / 'Cargo.lock',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               old, new]
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    watched += [source.resolve() for source in args.candidate_source]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'watched_sha256': before, 'runs': [], 'batches': list(args.batches), 'warmups': 5, 'trials': 30,
                'policy': 'process-local 4096 MiB cache; output drop and GPU barrier included',
                'scope': 'single-process exploratory old/new comparison; no Python gate',
                'limitations': ['CPU activity sampled every two seconds.',
                                ('GPU registry snapshots are system-wide, not per-process GPU time; '
                                 'sampling can perturb timing.' if args.gpu_activity else 'GPU activity not measured.'),
                                'No exclusive reservation or clock/power changes.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop, errors = threading.Event(), []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    thread.start()
    gpu_errors = []
    gpu_thread = None
    gpu_ready = threading.Event()
    if args.gpu_activity:
        gpu_thread = threading.Thread(target=monitor_gpu, args=(stop, out / 'gpu-activity.jsonl', gpu_errors, gpu_ready))
        gpu_thread.start()
    try:
        if args.gpu_activity and (not gpu_ready.wait(15) or gpu_errors):
            raise RuntimeError('GPU activity observer failed to start')
        for batch, order in [(32, [('baseline', old), ('candidate', new)]),
                             (128, [('candidate', new), ('baseline', old)])]:
            if batch not in args.batches:
                continue
            for label, binary in order:
                if gpu_errors:
                    raise RuntimeError('GPU activity observer failed')
                stem = f'{label}-b{batch}'
                command = [str(binary), str(ARTIFACT), str(batch), '4096']
                manifest['runs'].append({'name': stem, 'command': command})
                (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                run(command, stem, out)
                report = out / f'{stem}.json'
                report.write_bytes((out / f'{stem}.stdout').read_bytes())
                data = json.loads(report.read_bytes())
                trials = data['trials_seconds']
                if (data['batch'] != batch or data['warmups'] != 5 or len(trials) != 30
                        or data['cache_limit_bytes'] != 4096 * 1024 * 1024
                        or any(not math.isfinite(t) or t <= 0 for t in trials)):
                    raise ValueError(f'invalid result: {stem}')
                sorted_trials = sorted(trials)
                manifest['runs'][-1].update({'median_ms': statistics.median(trials) * 1000,
                                             'p95_ms': sorted_trials[28] * 1000,
                                             'maximum_ms': sorted_trials[-1] * 1000})
                (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                print(stem, 'complete', flush=True)
    finally:
        stop.set()
        thread.join(timeout=15)
        if gpu_thread is not None:
            gpu_thread.join(timeout=15)
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        manifest['gpu_telemetry_errors'] = gpu_errors
        manifest['gpu_activity_enabled'] = args.gpu_activity
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if (errors or gpu_errors or thread.is_alive() or (gpu_thread is not None and gpu_thread.is_alive())
            or not manifest['stable_inputs_and_sources']):
        raise RuntimeError('incomplete telemetry or changed source')
    print('exploratory comparison complete', flush=True)


if __name__ == '__main__':
    main()
