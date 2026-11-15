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

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'artifacts/canonical/tinystories-phase4-v1'
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
BINARY = ROOT / 'target/release/examples/tinystories_sample_bench'
BATCHES = (1, 4, 32, 128)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True,
                        help='native Metal comparison against the adapted fixed-code trace')
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise SystemExit('instrumented environment is not permitted')
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
               ROOT / 'tools/bench/phase3_process.py', ROOT / 'tools/reference/phase4_tinystories_mlx.py',
               ROOT / 'tools/reference/phase3_mlx_generator.py', ROOT / 'tools/reference/mlx_codec.py',
               ROOT / 'tools/reference/mlx_codec_assets.py', ROOT / 'tools/reference/mlx_operators.py',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               BINARY, qualification, core, core.parent / 'lib/libmlx.dylib', core.parent / 'lib/mlx.metallib',
               Path(packages['python']), ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'watched_sha256': before, 'qualification_sha256': args.qualification_sha256,
                'platform': platform.platform(), 'python_packages': packages,
                'warmups': 5, 'trials': 30, 'repetitions': 3, 'batches': BATCHES,
                'scope': 'generation plus synchronized code/ID host readback; excludes load',
                'sampling': 'different Rust SplitMix64 and Python MLX random streams',
                'limitations': ['CPU activity sampled every two seconds; GPU activity is not measured.',
                                'No exclusive reservation, CPU affinity or GPU clock/power changes.',
                                'RSS includes process startup and model loading.'], 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    errors = []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    thread.start()
    try:
        for j, batch in enumerate(BATCHES):
            for rep in range(3):
                for runtime in (('rust', 'python') if (j + rep) % 2 == 0 else ('python', 'rust')):
                    stem = f'{runtime}-b{batch}-r{rep}'
                    report = out / f'{stem}.json'
                    if runtime == 'rust':
                        command = [str(BINARY), str(ARTIFACT), str(batch)]
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
        for batch in BATCHES:
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
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if errors or thread.is_alive() or not manifest['stable_inputs_and_sources']:
        raise RuntimeError('incomplete telemetry or changed source')
    print('MLX comparison complete', flush=True)


if __name__ == '__main__':
    main()
