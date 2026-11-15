"""Retain a focused, time-bracketed batch-128 MLX tail diagnostic."""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path

from run_phase3_mlx_bench import digest, monitor, run

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'artifacts/canonical/tinystories-phase4-v1'
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
BINARY = ROOT / 'target/release/examples/tinystories_lifecycle_bench'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise ValueError('instrumented environment is not permitted')
    out = args.output.resolve()
    if out.exists():
        raise ValueError('output must be new')
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
               ROOT / 'tools/bench/phase3_process.py', ROOT / 'tools/reference/phase4_tinystories_mlx.py',
               ROOT / 'tools/reference/phase3_mlx_generator.py', ROOT / 'tools/reference/mlx_codec.py',
               ROOT / 'tools/reference/mlx_codec_assets.py', ROOT / 'tools/reference/mlx_operators.py',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               BINARY, ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    packages = json.loads(subprocess.check_output(
        [str(PYTHON), '-c', 'import importlib.metadata as m,importlib.util as u,json,pathlib,sys;'
         'p=pathlib.Path(u.find_spec("mlx.core").origin);'
         'print(json.dumps({"core":str(p),"python":str(pathlib.Path(sys.executable).resolve()),'
         '"mlx":m.version("mlx"),"mlx_metal":m.version("mlx-metal"),"numpy":m.version("numpy")}))'],
        cwd=ROOT, text=True))
    core = Path(packages['core'])
    watched += [core, core.parent / 'lib/libmlx.dylib', core.parent / 'lib/mlx.metallib', Path(packages['python'])]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'watched_sha256': before, 'python_packages': packages, 'batch': 128, 'pairs': 2, 'warmups': 5, 'trials': 30,
                'policy': 'Rust output cleanup/barrier included; default cache for both runtimes',
                'scope': 'diagnostic; does not replace the prior three-process gate',
                'limitations': ['CPU activity sampled every two seconds; GPU activity not measured.',
                                'No exclusive reservation or GPU clock/power changes.'], 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    errors = []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    thread.start()
    try:
        for repeat in range(2):
            for runtime in (('rust', 'python') if repeat == 0 else ('python', 'rust')):
                stem = f'{runtime}-b128-r{repeat}'
                report = out / f'{stem}.json'
                if runtime == 'rust':
                    command = [str(BINARY), str(ARTIFACT), '128', 'default']
                else:
                    command = [str(PYTHON), str(ROOT / 'tools/reference/phase4_tinystories_mlx.py'),
                               '--artifact', str(ARTIFACT), '--batch', '128', '--bench', '--output', str(report)]
                entry = {'name': stem, 'command': command, 'start_monotonic': time.monotonic()}
                manifest['runs'].append(entry)
                (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                run(command, stem, out)
                entry['end_monotonic'] = time.monotonic()
                if runtime == 'rust':
                    report.write_bytes((out / f'{stem}.stdout').read_bytes())
                data = json.loads(report.read_bytes())
                trials = data['trials_seconds']
                if data['batch'] != 128 or data['warmups'] != 5 or len(trials) != 30 or any(not math.isfinite(x) or x <= 0 for x in trials):
                    raise ValueError(f'invalid measurement: {stem}')
                entry['median_ms'] = statistics.median(trials) * 1000
                entry['max_ms'] = max(trials) * 1000
                entry['trials_over_4_seconds'] = sum(t > 4 for t in trials)
                (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                print(stem, 'complete', flush=True)
    finally:
        stop.set()
        thread.join(timeout=15)
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if errors or thread.is_alive() or not manifest['stable_inputs_and_sources']:
        raise RuntimeError('incomplete telemetry or changed source')
    print('diagnostic complete', flush=True)


if __name__ == '__main__':
    main()
