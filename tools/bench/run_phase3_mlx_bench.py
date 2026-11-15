"""Qualify and compare fresh Rust/Python MLX Sudoku processes on Apple Silicon."""
import argparse
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/canonical/sudoku-data-phase3'
ARTIFACT = ROOT / 'artifacts/canonical/sudoku-easy-qualified'
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
BINARY = ROOT / 'target/release/examples/mlx_generator_bench'
CLI = ROOT / 'target/release/ncm'
BATCHES = (1, 4, 32, 128)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, stem, output):
    wrapper = [sys.executable, str(ROOT / 'tools/bench/phase3_process.py'),
               '--stdout', str(output / (stem + '.stdout')),
               '--stderr', str(output / (stem + '.stderr')),
               '--metadata', str(output / (stem + '.meta.json')), '--', *command]
    process = subprocess.Popen(wrapper, cwd=ROOT, start_new_session=True)
    try:
        code = process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(f'{stem} timed out')
    if code:
        raise RuntimeError(f'{stem} exited {code}; see retained stderr')


def monitor(stop, path, errors):
    try:
        with path.open('x') as out:
            while not stop.is_set():
                result = subprocess.run(['ps', '-Ao', 'pid=,pcpu=,comm='], text=True,
                                        capture_output=True, timeout=10, check=True)
                entries = []
                for line in result.stdout.splitlines():
                    pid, cpu, name = line.split(None, 2)
                    entries.append({'pid': int(pid), 'cpu_percent': float(cpu), 'name': Path(name).name})
                out.write(json.dumps({'monotonic': time.monotonic(), 'load_average': os.getloadavg(),
                                      'top_cpu': sorted(entries, key=lambda row: row['cpu_percent'], reverse=True)[:12]}) + '\n')
                out.flush()
                stop.wait(2)
    except Exception as error:
        errors.append(str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='full pinned Python CUDA easy result')
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise SystemExit('instrumented environment is not permitted')
    reference = args.reference.resolve()
    if digest(reference) != args.reference_sha256:
        raise SystemExit('reference SHA-256 mismatch')
    ref = json.loads(reference.read_bytes())
    if ref.get('difficulty') != 'easy' or ref.get('upstream') != 'c70274273f489abc73832b2b98584a8483b6b203':
        raise ValueError('wrong original Python reference')
    if len(ref['codes']) != 2000 or len(ref['ids']) != 2000:
        raise ValueError('expected full original Python reference')
    out = args.output.resolve()
    if out.exists(): raise SystemExit('output exists; choose a new run identity')
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/phase3_process.py',
               ROOT / 'tools/reference/phase3_mlx_generator.py', ROOT / 'tools/reference/mlx_codec.py',
               ROOT / 'tools/reference/mlx_codec_assets.py', ROOT / 'tools/reference/mlx_operators.py',
               ROOT / 'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               BINARY, CLI, reference]
    package_info = json.loads(subprocess.check_output(
        [str(PYTHON), '-c',
         'import importlib.metadata as m,importlib.util as u,json,pathlib,sys;'
         'p=pathlib.Path(u.find_spec("mlx.core").origin);'
         'print(json.dumps({"core":str(p),"python":str(pathlib.Path(sys.executable).resolve()),'
         '"mlx":m.version("mlx"),"mlx_metal":m.version("mlx-metal"),"numpy":m.version("numpy")}))'],
        cwd=ROOT, text=True))
    core = Path(package_info['core'])
    watched += [core, core.parent / 'lib/libmlx.dylib', core.parent / 'lib/mlx.metallib', Path(package_info['python'])]
    watched += list((ROOT / 'crates').rglob('*.rs'))
    watched += list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    watched += [DATA / f'easy-{batch}-inputs.json' for batch in BATCHES]
    watched.append(DATA / 'easy-2000-inputs.json')
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'watched_sha256': before, 'reference_sha256': args.reference_sha256,
                'platform': platform.platform(), 'processor': platform.processor(), 'python_packages': package_info,
                'warmups': 5, 'trials': 30, 'repetitions': 3, 'batches': BATCHES,
                'limitations': ['CPU activity sampled every two seconds; GPU activity is not measured.',
                                'No exclusive reservation, CPU affinity or GPU clock/power changes.',
                                'RSS includes process startup and model loading.'], 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event(); errors = []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors)); thread.start()
    try:
        inputs128 = DATA / 'easy-128-inputs.json'
        expected_inputs = json.loads(inputs128.read_bytes())
        # The qualified reference records the SHA of all 2,000 inputs. The
        # 128-board subset must also match the separately stored evaluation.
        full = json.loads((DATA / 'easy-2000-inputs.json').read_bytes())
        if hashlib.sha256((DATA / 'easy-2000-inputs.json').read_bytes()).hexdigest() != ref['inputs_sha256'] or expected_inputs != full[:128]:
            raise ValueError('qualification inputs differ from original reference')
        py_result = out / 'qualification-python.json'
        run([str(PYTHON), str(ROOT / 'tools/reference/phase3_mlx_generator.py'), '--artifact', str(ARTIFACT),
             '--inputs', str(inputs128), '--output', str(py_result)], 'qualification-python', out)
        run([str(CLI), 'generate', str(ARTIFACT), 'mlx', str(inputs128), '128'], 'qualification-rust', out)
        py = json.loads(py_result.read_bytes())
        rust = json.loads((out / 'qualification-rust.stdout').read_bytes())
        qualification = {f'{runtime}_{field}_matching_rows': sum(a == b for a, b in zip(value[field], ref[field][:128], strict=True))
                         for runtime, value in (('python', py), ('rust', rust)) for field in ('codes', 'ids')}
        (out / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
        if any(count != 128 for count in qualification.values()):
            raise ValueError('128-board reference qualification failed; retained evidence')
        for j, batch in enumerate(BATCHES):
            for repeat in range(3):
                for runtime in (('rust', 'python') if (j + repeat) % 2 == 0 else ('python', 'rust')):
                    stem = f'{runtime}-b{batch}-r{repeat}'
                    inputs = DATA / f'easy-{batch}-inputs.json'
                    report = out / (stem + '.json')
                    if runtime == 'rust':
                        command = [str(BINARY), str(ARTIFACT), str(inputs), str(batch)]
                    else:
                        command = [str(PYTHON), str(ROOT / 'tools/reference/phase3_mlx_generator.py'),
                                   '--artifact', str(ARTIFACT), '--inputs', str(inputs), '--output', str(report), '--bench']
                    manifest['runs'].append({'name': stem, 'command': command})
                    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    run(command, stem, out)
                    if runtime == 'rust': report.write_bytes((out / (stem + '.stdout')).read_bytes())
                    result = json.loads(report.read_bytes())
                    if result['batch'] != batch or result['warmups'] != 5 or len(result['trials']) != 30 or any(not math.isfinite(x) or x <= 0 for x in result['trials']):
                        raise ValueError(f'incomplete measurement: {stem}')
                    print(stem, 'complete', flush=True)
        summary = {}
        for batch in BATCHES:
            hashes = set(); values = {}
            for runtime in ('rust', 'python'):
                reports = [json.loads((out / f'{runtime}-b{batch}-r{i}.json').read_bytes()) for i in range(3)]
                metas = [json.loads((out / f'{runtime}-b{batch}-r{i}.meta.json').read_bytes()) for i in range(3)]
                hashes.update(report['outputs_sha256_i64le'] for report in reports)
                values[runtime] = {'median_ms': statistics.median(x for report in reports for x in report['trials']) * 1000,
                                   'process_medians_ms': [statistics.median(r['trials']) * 1000 for r in reports],
                                   'peak_rss_bytes': [m['child_peak_rss_platform_units'] for m in metas],
                                   'mlx_memory': [r.get('mlx_memory', {'peak': r.get('mlx_peak_bytes'),
                                                  'active': r.get('mlx_active_bytes'), 'cache': r.get('mlx_cache_bytes')}) for r in reports]}
            if len(hashes) != 1: raise ValueError(f'output mismatch during benchmark at batch {batch}')
            values['rust_over_python_median'] = values['rust']['median_ms'] / values['python']['median_ms']
            summary[str(batch)] = values
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        stop.set(); thread.join(timeout=15)
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if errors or thread.is_alive() or not manifest['stable_inputs_and_sources']:
        raise RuntimeError('incomplete telemetry or changed source')
    print('MLX comparison complete', flush=True)


if __name__ == '__main__':
    main()
