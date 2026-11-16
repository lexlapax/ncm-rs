"""Interleave byte-identified XSum binaries to investigate a latency change."""
import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from run_phase4_xsum_bench import linked_libraries, monitor, terminate_group

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'reports/phase4-xsum-real-inputs-v1/prefixes.json'
ARTIFACT = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
FILES = ROOT / 'reports/phase4-xsum-tie-paired-regression-v1'
EXPECTED = {'baseline-binary': '95ad92a52ae7f1d1353293517404d50eecbe9b1283eb6afe5d8e68f1034b7e86',
            'current-binary': 'e8d68bb0619f0f0041130e14528cc9ac695272644db2924ef88a298bd90e63d5'}


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def cpu_monitor(stop, path):
    with path.open('x') as output:
        while not stop.is_set():
            sample = {'monotonic': time.monotonic(), 'load': os.getloadavg()}
            try:
                process = subprocess.run(['ps', '-eo', 'pid,pcpu,comm', '--sort=-pcpu'],
                                         capture_output=True, text=True, timeout=5, check=False)
                sample['status'] = process.returncode
                sample['top'] = process.stdout.splitlines()[:9]
            except (OSError, subprocess.TimeoutExpired) as error:
                sample['error'] = str(error)
            output.write(json.dumps(sample) + '\n'); output.flush()
            stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.output.is_symlink() or not out.is_relative_to((ROOT / 'reports').resolve()) or out.exists():
        raise ValueError('new report directory under reports required')
    if any(os.environ.get(name) for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH',
           'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
           'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
           'NCM_PHASE3_LT_PROFILE')):
        raise ValueError('environment overrides are unsupported')
    watched = {str(FILES / name): sha for name, sha in EXPECTED.items()}
    watched[str(INPUTS)] = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'
    watched[str(ARTIFACT / 'manifest.json')] = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
    if any(digest(Path(path)) != sha for path, sha in watched.items()):
        raise ValueError('binary, input or artifact identity differs')
    libraries = {name: linked_libraries(FILES / f'{name}-binary', CUDNN)
                 for name in ('baseline', 'current')}
    if libraries['baseline'] != libraries['current']:
        raise ValueError('binary native libraries differ')
    out.mkdir(parents=True)
    manifest = {'schema_version': 1, 'baseline_commit': '531ea96', 'tie_commit': '902344f',
                'watched_sha256': watched, 'batch_trials': {'1': 30, '4': 30, '32': 5},
                'processes_per_binary_batch': 3, 'order': 'alternating paired binaries',
                'scope': 'generation and code/ID readback, excludes load',
                'baseline_binary_reproduces_previous_benchmark': True,
                'current_binary_reproduces_tie_benchmark': True,
                'runner_sha256': digest(Path(__file__)), 'native_libraries': libraries,
                'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                                '--format=csv,noheader'], text=True).strip()}
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    stop = threading.Event()
    threads = [threading.Thread(target=monitor, args=(stop, out / 'gpu-activity.jsonl'), daemon=True),
               threading.Thread(target=cpu_monitor, args=(stop, out / 'cpu-activity.jsonl'), daemon=True)]
    for thread in threads:
        thread.start()
    process = None
    signatures = {}
    try:
        for number, batch in enumerate((1, 4, 32)):
            for rep in range(3):
                order = ('baseline', 'current') if (number + rep) % 2 == 0 else ('current', 'baseline')
                for runtime in order:
                    name = f'{runtime}-b{batch}-r{rep}'
                    command = [str(FILES / f'{runtime}-binary'), str(ARTIFACT), str(INPUTS), str(batch), 'device']
                    if batch != 32:
                        command.append('30')
                    env = os.environ.copy(); env['LD_LIBRARY_PATH'] = str(CUDNN); env['OMP_NUM_THREADS'] = '1'
                    with (out / f'{name}.stdout').open('xb') as stdout, (out / f'{name}.stderr').open('xb') as stderr:
                        process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(out / f'{name}.time'), *command],
                                                   cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                        with (out / 'process-events.jsonl').open('a') as events:
                            events.write(json.dumps({'event': 'start', 'name': name, 'pid': process.pid,
                                                     'monotonic': time.monotonic()}) + '\n')
                        try:
                            code = process.wait(timeout=600)
                        except subprocess.TimeoutExpired:
                            terminate_group(process); raise RuntimeError(f'{name} timed out')
                    with (out / 'process-events.jsonl').open('a') as events:
                        events.write(json.dumps({'event': 'end', 'name': name, 'returncode': code,
                                                 'monotonic': time.monotonic()}) + '\n')
                    process = None
                    if code:
                        raise RuntimeError(f'{name} exited {code}')
                    result = json.loads((out / f'{name}.stdout').read_text().splitlines()[-1])
                    trials = 5 if batch == 32 else 30
                    if result['batch'] != batch or len(result['trials_seconds']) != trials:
                        raise ValueError('incomplete timing result')
                    signature = (result['codes'], result['ids'])
                    if batch in signatures and signatures[batch] != signature:
                        raise ValueError('paired binary outputs differ')
                    signatures[batch] = signature
                    (out / f'{name}.json').write_text(json.dumps(result, sort_keys=True) + '\n')
                    print(name, 'complete', flush=True)
    finally:
        if process is not None:
            terminate_group(process)
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
    if any(digest(Path(path)) != sha for path, sha in watched.items()):
        raise ValueError('watched file changed during comparison')
    if any(linked_libraries(FILES / f'{name}-binary', CUDNN) != libraries[name]
           for name in ('baseline', 'current')):
        raise ValueError('native libraries changed during comparison')
    print('complete', flush=True)


if __name__ == '__main__':
    main()
