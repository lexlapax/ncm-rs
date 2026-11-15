"""One paired full-codec diagnostic; never an acceptance gate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / '.reference/phase2-mlx-linear-fused-codec-v1'
EVIDENCE = ROOT / 'reports/phase2-mlx-linear-fused-codec-v1'
OUT = ROOT / 'reports/phase2-mlx-linear-fused-codec-pairs-v1'
BINARIES = {
    'current': ROOT / 'target/release/examples/mlx_codec_bench',
    'fused': ROOT / '.reference/phase2-mlx-linear-fused-codec-target-v1/release/examples/mlx_codec_bench',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    qual = json.loads((EVIDENCE / 'qualification.json').read_bytes())
    if qual['all_passed'] is not True or qual['trace_count'] != 642:
        raise ValueError('trained qualification did not pass')
    trace = EVIDENCE / 'trace'
    if sha(trace / 'report.json') != qual['candidate_report_sha256']:
        raise ValueError('qualified trace report changed')
    trace_report = json.loads((trace / 'report.json').read_bytes())
    if sha(trace / 'traces.safetensors') != trace_report['trace_sha256']:
        raise ValueError('qualified trace bytes changed')
    if sha(EXPERIMENT / 'crates/ncm-backend-mlx/src/ops.rs') != trace_report['source_sha256']['crates/ncm-backend-mlx/src/ops.rs']:
        raise ValueError('qualified operator source changed')
    matrix = json.loads((ROOT / 'reports/phase2-mlx-gate18-matrix-v1/environment.json').read_bytes())
    if sha(BINARIES['current']) != matrix['binary_sha256']['candidate']['mlx_codec_bench']:
        raise ValueError('current executable differs from passing control')
    if OUT.exists():
        raise ValueError('paired output already exists')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact', 'LD_PRELOAD')):
        raise ValueError('instrumented environment')
    OUT.mkdir(mode=0o700)
    meta = {'status': 'running', 'scope': 'paired full-codec diagnostic, not acceptance',
            'qualification_sha256': sha(EVIDENCE / 'qualification.json'),
            'binary_sha256': {k: sha(v) for k, v in BINARIES.items()},
            'patch_sha256': sha(ROOT / 'docs/results/2026-11-11-phase2-optimization/mlx-linear-fused-codec-v1.patch'),
            'source_sha256': {k: sha((ROOT if k == 'current' else EXPERIMENT) / 'crates/ncm-backend-mlx/src/ops.rs') for k in BINARIES},
            'native_sha256': {name: sha(ROOT / '.reference/mlx-native/lib' / name) for name in ('libmlx.dylib', 'mlx.metallib')},
            'thermal_before': subprocess.check_output(['/usr/bin/pmset', '-g', 'therm'], text=True), 'commands': []}
    stop = threading.Event()
    def save():
        (OUT / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    def sample():
        try:
            with (OUT / 'activity.jsonl').open('x') as file:
                while not stop.is_set():
                    entries = []
                    for line in subprocess.check_output(['/bin/ps', '-Ao', 'pcpu=,comm='], text=True).splitlines():
                        cpu, name = line.split(None, 1)
                        entries.append({'cpu_percent': float(cpu), 'name': Path(name).name})
                    file.write(json.dumps({'time': time.time(), 'top_cpu': sorted(entries, key=lambda x: x['cpu_percent'], reverse=True)[:12]}) + '\n')
                    file.flush(); stop.wait(2)
        except Exception as exc:
            meta['activity_error'] = str(exc)
    save(); thread = threading.Thread(target=sample); thread.start()
    try:
        for repeat in range(1, 4):
            for name in (('current', 'fused') if repeat % 2 else ('fused', 'current')):
                report = OUT / f'{name}-{repeat}.json'
                command = [str(BINARIES[name]), str(ROOT / 'artifacts/canonical/sudoku-easy-qualified'),
                           'tests/fixtures/codec-bench-v1.json', str(report)]
                cwd = ROOT if name == 'current' else EXPERIMENT
                entry = {'name': name, 'repeat': repeat, 'argv': command, 'cwd': str(cwd), 'start': time.time()}
                meta['commands'].append(entry); save()
                with (OUT / f'{name}-{repeat}.stdout').open('xb') as stdout, (OUT / f'{name}-{repeat}.stderr').open('xb') as stderr:
                    child = subprocess.run(command, cwd=cwd, env={'PATH': '/usr/bin:/bin'}, stdout=stdout, stderr=stderr, timeout=120, check=False)
                entry['end'] = time.time(); entry['exit_code'] = child.returncode; save()
                if child.returncode != 0:
                    raise RuntimeError(f'{name} {repeat} failed')
                entry['report_sha256'] = sha(report); save()
    except Exception as exc:
        meta['status'] = 'failed'; meta['error'] = str(exc); raise
    finally:
        stop.set(); thread.join()
        meta['thermal_after'] = subprocess.check_output(['/usr/bin/pmset', '-g', 'therm'], text=True); save()
    if 'activity_error' in meta:
        meta['status'] = 'failed'; save(); raise RuntimeError(meta['activity_error'])
    meta['status'] = 'complete'; save(); print('paired full-codec processes complete')

if __name__ == '__main__':
    main()
