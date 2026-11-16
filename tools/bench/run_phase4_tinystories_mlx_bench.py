"""Compare qualified TinyStories Rust and Python MLX in fresh alternating processes."""
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
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
BINARY = ROOT / 'target/release/examples/tinystories_sample_bench'
BATCHES = (1, 4, 32, 128)


def runtime_order(batch, repetition):
    return ('rust', 'python') if (BATCHES.index(batch) + repetition) % 2 == 0 else ('python', 'rust')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True,
                        help='native Metal comparison against the adapted fixed-code trace')
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, help='explicit Rust generation benchmark binary')
    parser.add_argument('--candidate-source-root', type=Path, help='isolated candidate source tree to fingerprint')
    parser.add_argument('--rust-lifecycle-policy', action='store_true',
                        help='include output cleanup/barrier; use a process-local 512 MiB cache at batches 1 and 4')
    parser.add_argument('--batches', type=int, nargs='+', choices=BATCHES, default=BATCHES,
                        help='batches to measure; default is all four')
    parser.add_argument('--rust-large-cache-mib', type=int,
                        help='explicit Rust process cache limit for batch 32 and 128')
    parser.add_argument('--gpu-activity', action='store_true',
                        help='sample system-wide GPU registry counters; diagnostic timings may be perturbed')
    args = parser.parse_args()
    if len(set(args.batches)) != len(args.batches):
        parser.error('batches must be unique')
    if args.rust_large_cache_mib is not None and (not args.rust_lifecycle_policy
                                                  or not 1 <= args.rust_large_cache_mib <= 16384):
        parser.error('large cache limit requires lifecycle policy and must be 1..=16384 MiB')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise SystemExit('instrumented environment is not permitted')
    binary = (ROOT / 'target/release/examples/tinystories_lifecycle_bench'
              if args.rust_lifecycle_policy else BINARY)
    if args.binary:
        binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError('Rust benchmark binary must be an existing file')
    qualification = args.qualification.resolve()
    if digest(qualification) != args.qualification_sha256:
        raise ValueError('qualification digest differs')
    stats = json.loads(qualification.read_bytes())
    heads = [f'generator.level_{level:02d}.logits' for level in range(1, 12)]
    for name in heads + ['generator.final_logits']:
        row = stats[name]
        if row['phase2_accumulated_band_atol_5e-4_rtol_2e-4_outside'] != 0:
            raise ValueError(f'unchanged accumulated comparison band failed: {name}')
        if name in heads and row['argmax_differences'] != 0:
            raise ValueError(f'head argmax differs: {name}')
    if stats['generator.output_ids']['differences'] != 0:
        raise ValueError('final IDs differ from adapted reference')
    out = args.output.resolve()
    if out.exists():
        raise ValueError('output must be new')
    packages = json.loads(subprocess.check_output(
        [str(PYTHON), '-c', 'import importlib.metadata as m,importlib.util as u,json,pathlib,sys;'
         'p=pathlib.Path(u.find_spec("mlx.core").origin);'
         'print(json.dumps({"core":str(p),"python":str(pathlib.Path(sys.executable).resolve()),'
         '"mlx":m.version("mlx"),"mlx_metal":m.version("mlx-metal"),"numpy":m.version("numpy")}))'],
        cwd=ROOT, text=True))
    core = Path(packages['core'])
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
               ROOT / 'tools/bench/phase3_process.py', ROOT / 'tools/bench/phase4_mlx_gpu_activity.py',
               ROOT / 'tools/reference/phase4_tinystories_mlx.py',
               ROOT / 'tools/reference/phase3_mlx_generator.py', ROOT / 'tools/reference/mlx_codec.py',
               ROOT / 'tools/reference/mlx_codec_assets.py', ROOT / 'tools/reference/mlx_operators.py',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               binary, qualification, core, core.parent / 'lib/libmlx.dylib', core.parent / 'lib/mlx.metallib',
               Path(packages['python']), ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    if args.candidate_source_root:
        source_root = args.candidate_source_root.resolve()
        if not (source_root / 'crates').is_dir():
            raise ValueError('candidate source tree must contain crates')
        watched += [source_root / 'Cargo.toml', source_root / 'Cargo.lock']
        watched += list((source_root / 'crates').rglob('*.rs')) + list((source_root / 'crates').rglob('Cargo.toml'))
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'watched_sha256': before, 'qualification_sha256': args.qualification_sha256,
                'platform': platform.platform(), 'python_packages': packages,
                'warmups': 5, 'trials': 30, 'repetitions': 3, 'batches': args.batches,
                'scope': 'generation plus synchronized code/ID host readback; excludes load',
                'sampling': 'different Rust SplitMix64 and Python MLX random streams',
                'rust_policy': ('drop unused logits and evaluate GPU barrier within timing; '
                                'process-local 512 MiB cache for batch 1/4; batch 32/128: ' +
                                (f'{args.rust_large_cache_mib} MiB' if args.rust_large_cache_mib is not None
                                 else 'default cache')
                                if args.rust_lifecycle_policy else 'default cache and output lifetime'),
                'limitations': ['CPU activity sampled every two seconds.',
                                ('GPU registry snapshots are system-wide, not per-process GPU time; '
                                 'sampling can perturb timing.' if args.gpu_activity else 'GPU activity not measured.'),
                                'No exclusive reservation, CPU affinity or GPU clock/power changes.',
                                'RSS includes process startup and model loading.'], 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    errors = []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    thread.start()
    gpu_errors = []
    gpu_ready = threading.Event()
    gpu_thread = None
    if args.gpu_activity:
        gpu_thread = threading.Thread(target=monitor_gpu, args=(stop, out / 'gpu-activity.jsonl', gpu_errors, gpu_ready))
        gpu_thread.start()
    try:
        if args.gpu_activity and (not gpu_ready.wait(15) or gpu_errors):
            raise RuntimeError('GPU activity observer failed to start')
        for batch in args.batches:
            for rep in range(3):
                for runtime in runtime_order(batch, rep):
                    if gpu_errors:
                        raise RuntimeError('GPU activity observer failed')
                    stem = f'{runtime}-b{batch}-r{rep}'
                    report = out / f'{stem}.json'
                    if runtime == 'rust':
                        command = [str(binary), str(ARTIFACT), str(batch)]
                        if args.rust_lifecycle_policy:
                            command.append('512' if batch in (1, 4) else
                                           str(args.rust_large_cache_mib) if args.rust_large_cache_mib is not None
                                           else 'default')
                    else:
                        command = [str(PYTHON), str(ROOT / 'tools/reference/phase4_tinystories_mlx.py'),
                                   '--artifact', str(ARTIFACT), '--batch', str(batch), '--bench', '--output', str(report)]
                    manifest['runs'].append({'name': stem, 'command': command})
                    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    run(command, stem, out)
                    if runtime == 'rust':
                        report.write_bytes((out / f'{stem}.stdout').read_bytes())
                    result = json.loads(report.read_bytes())
                    values = result['trials_seconds']
                    if (result['batch'] != batch or result['warmups'] != 5 or len(values) != 30
                            or any(not math.isfinite(x) or x <= 0 for x in values)):
                        raise ValueError(f'incomplete measurement: {stem}')
                    print(stem, 'complete', flush=True)
        summary = {}
        for batch in args.batches:
            pair = {}
            for runtime in ('rust', 'python'):
                reports = [json.loads((out / f'{runtime}-b{batch}-r{i}.json').read_bytes()) for i in range(3)]
                meta = [json.loads((out / f'{runtime}-b{batch}-r{i}.meta.json').read_bytes()) for i in range(3)]
                trials = sorted(t for report in reports for t in report['trials_seconds'])
                medians = [statistics.median(report['trials_seconds']) * 1000 for report in reports]
                pair[runtime] = {'median_process_median_ms': statistics.median(medians), 'process_medians_ms': medians,
                                 'pooled_p95_ms': trials[math.ceil(.95 * len(trials)) - 1] * 1000,
                                 'peak_rss_bytes': [m['child_peak_rss_platform_units'] for m in meta],
                                 'mlx_memory': [r['mlx_memory'] for r in reports]}
            summary[str(batch)] = pair
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
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
    print('MLX comparison complete', flush=True)


if __name__ == '__main__':
    main()
