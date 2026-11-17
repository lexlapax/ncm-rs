"""Measure XSum Rust and adapted Python MLX in alternating fresh processes."""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import threading
from pathlib import Path

from phase4_mlx_gpu_activity import monitor_gpu
from run_phase3_mlx_bench import digest, monitor, run

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
INPUTS = ROOT / 'reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json'
REFERENCE = ROOT / 'reports/phase4-xsum-mlx-cli-inputs-v1/reference.json'
BINARY = ROOT / 'target/release/examples/xsum_perf_screen'
CLI = ROOT / 'target/release/ncm'
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
BATCHES = (1, 4, 32)
ARTIFACT_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
INPUT_SHA = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'
REFERENCE_SHA = 'b2a20dbad883417406387726d799c8f79ecf814507f58d01bd05204e16775125'


def flattened(value):
    if not isinstance(value, list):
        raise ValueError('output is not a list')
    return [item for row in value for item in row] if value and isinstance(value[0], list) else value


def checked_output(result, batch):
    if (result.get('batch') != batch or result.get('warmups') != 5 or
            len(result.get('trials_seconds', [])) != 30 or
            any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                for x in result['trials_seconds'])):
        raise ValueError('invalid timing record')
    codes, ids = flattened(result['codes']), flattened(result['ids'])
    if (len(codes) != batch * 200 or len(ids) != batch * 48 or
            any(type(x) is not int or not 0 <= x < 4096 for x in codes) or
            any(type(x) is not int or not 0 <= x < 50304 for x in ids)):
        raise ValueError('invalid output shape or values')
    return codes, ids


def order(batch, repetition):
    return ('rust', 'python') if (BATCHES.index(batch) + repetition) % 2 == 0 else ('python', 'rust')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-activity', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise ValueError('instrumented environment is unsupported')
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    qualification = args.qualification.resolve()
    if args.qualification.is_symlink() or digest(qualification) != args.qualification_sha256:
        raise ValueError('qualification identity differs')
    result = json.loads(qualification.read_bytes())
    qualified_sources = result.get('source_sha256', {})
    if (not result.get('shapes_ok') or result.get('code_differences') != 0 or result.get('id_differences') != 0 or
            qualified_sources.get(str(CLI)) != digest(CLI) or
            qualified_sources.get(str(ARTIFACT / 'manifest.json')) != ARTIFACT_SHA or
            qualified_sources.get(str(INPUTS)) != INPUT_SHA or qualified_sources.get(str(REFERENCE)) != REFERENCE_SHA):
        raise ValueError('candidate CLI qualification differs')
    if (digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA or digest(INPUTS) != INPUT_SHA or
            digest(REFERENCE) != REFERENCE_SHA):
        raise ValueError('artifact or input identity differs')
    packages = json.loads(subprocess.check_output(
        [str(PYTHON), '-c', 'import importlib.metadata as m,importlib.util as u,json,pathlib,sys;'
         'p=pathlib.Path(u.find_spec("mlx.core").origin);'
         'print(json.dumps({"core":str(p),"python":str(pathlib.Path(sys.executable).resolve()),'
         '"mlx":m.version("mlx"),"mlx_metal":m.version("mlx-metal"),"numpy":m.version("numpy")}))'],
        cwd=ROOT, text=True))
    core = Path(packages['core'])
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
               ROOT / 'tools/bench/phase3_process.py', ROOT / 'tools/bench/phase4_mlx_gpu_activity.py',
               ROOT / 'tools/reference/phase4_xsum_mlx.py', ROOT / 'tools/reference/phase3_mlx_generator.py',
               ROOT / 'tools/reference/mlx_codec.py', ROOT / 'tools/reference/mlx_codec_assets.py',
               ROOT / 'tools/reference/mlx_operators.py', ROOT / 'tools/reference/xsum_numpy_selection.py',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               BINARY, CLI, INPUTS, REFERENCE, qualification, core, core.parent / 'lib/libmlx.dylib',
               core.parent / 'lib/mlx.metallib', Path(packages['python']), ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'watched_sha256': before, 'qualification_sha256': args.qualification_sha256,
                'platform': platform.platform(), 'python_packages': packages,
                'warmups': 5, 'trials': 30, 'repetitions': 3, 'batches': BATCHES,
                'scope': 'generation and synchronized output readback; model load excluded from timing',
                'limitations': ['RSS includes process startup and load.', 'No CPU affinity or exclusive GPU reservation.',
                                'Python MLX is an adapted reference, not original PyTorch on Metal.',
                                'GPU registry sampling is system-wide and can perturb timing.' if args.gpu_activity
                                else 'GPU activity not continuously observed.'], 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop, errors, gpu_errors, ready = threading.Event(), [], [], threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    thread.start()
    gpu_thread = None
    if args.gpu_activity:
        gpu_thread = threading.Thread(target=monitor_gpu, args=(stop, out / 'gpu-activity.jsonl', gpu_errors, ready))
        gpu_thread.start()
    try:
        if args.gpu_activity and (not ready.wait(15) or gpu_errors):
            raise RuntimeError('GPU observer did not start')
        signatures = {}
        for batch in BATCHES:
            for rep in range(3):
                for runtime in order(batch, rep):
                    if errors or gpu_errors:
                        raise RuntimeError('activity observer failed')
                    stem = f'{runtime}-b{batch}-r{rep}'
                    report = out / f'{stem}.json'
                    command = ([str(BINARY), str(ARTIFACT), str(INPUTS), str(batch), '5', '30']
                               if runtime == 'rust' else
                               [str(PYTHON), str(ROOT / 'tools/reference/phase4_xsum_mlx.py'), '--artifact', str(ARTIFACT),
                                '--inputs', str(INPUTS), '--batch', str(batch), '--bench', '--warmups', '5',
                                '--trials', '30', '--output', str(report)])
                    manifest['runs'].append({'name': stem, 'command': command})
                    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    run(command, stem, out)
                    if runtime == 'rust':
                        report.write_bytes((out / f'{stem}.stdout').read_bytes())
                    signature = checked_output(json.loads(report.read_bytes()), batch)
                    if batch in signatures and signatures[batch] != signature:
                        raise ValueError(f'output changed: {stem}')
                    signatures[batch] = signature
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
        if gpu_thread:
            gpu_thread.join(timeout=15)
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        manifest['gpu_telemetry_errors'] = gpu_errors
        manifest['gpu_activity_enabled'] = args.gpu_activity
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if errors or gpu_errors or thread.is_alive() or (gpu_thread and gpu_thread.is_alive()) or not manifest['stable_inputs_and_sources']:
        raise RuntimeError('incomplete telemetry or changed source')
    print('MLX comparison complete', flush=True)


if __name__ == '__main__':
    main()
