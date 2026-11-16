"""Record short, sequential XSum capacity screens; OOMs remain visible."""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from run_phase4_xsum_bench import (ARTIFACT, BENCH, REFERENCE, ROOT, SOURCE, SOURCES,
                                   digest, monitor, terminate_group)

INPUTS = ROOT / 'reports/phase4-xsum-capacity-inputs-v1/prefixes.json'
INPUT_SHA = '96a215963a9acd6e7adb9af868a8d03ffd3110446bc3a5ac6434163a000345a2'
ARTIFACT_SHA = '5df37749e64418c586472a4678600603992357b9bc5e3611da307918e13b237e'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cudnn-dir', type=Path, required=True)
    ap.add_argument('--batches', type=int, nargs='+', default=[64, 128], choices=(64, 128))
    args = ap.parse_args()
    if len(set(args.batches)) != len(args.batches): raise ValueError('duplicate batches')
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_INJECTION64_PATH',
                 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG',
                 'NVIDIA_TF32_OVERRIDE', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
    if any(os.environ.get(k) for k in forbidden): raise ValueError('unsupported environment override')
    out = args.output.resolve()
    if args.output.is_symlink() or not out.is_relative_to((ROOT / 'reports').resolve()) or out.exists():
        raise ValueError('output must be a new directory under reports')
    cudnn = args.cudnn_dir.resolve()
    cudnn_files = sorted(cudnn.glob('libcudnn*.so.9'))
    if not (cudnn / 'libcudnn.so.9').is_file() or not BENCH.is_file() or not Path('/usr/bin/time').is_file():
        raise ValueError('missing strict CUDA binary, cuDNN or time')
    if digest(INPUTS) != INPUT_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA:
        raise ValueError('capacity inputs or artifact differ')
    paths = [ROOT / name for name in SOURCES] + [Path(__file__).resolve(), BENCH, INPUTS,
                                                     ARTIFACT / 'manifest.json', SOURCE / 'config.yaml']
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    cudnn_hashes = {p.name: digest(p) for p in cudnn_files}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                   '--format=csv,noheader'], text=True).strip().splitlines()
    if len(gpu) != 1: raise ValueError('exactly one GPU required')
    out.mkdir(parents=True)
    manifest = {'source_sha256': source_hashes, 'cudnn_sha256': cudnn_hashes,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'gpu': gpu[0],
                'batches': args.batches, 'warmups': 2, 'trials': 5,
                'scope': 'exploratory capacity screen; synchronized generation and code/ID readback',
                'python_cleanup': 'explicit garbage collection and synchronization timed separately',
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl'), daemon=True)
    thread.start()
    process = None
    results = []
    signatures = {}
    try:
        for batch in args.batches:
            for runtime in ('rust', 'python'):
                name = f'{runtime}-b{batch}'
                if runtime == 'rust':
                    command = [str(BENCH), str(ARTIFACT), str(INPUTS), str(batch), 'device']
                else:
                    command = [str(REFERENCE), str(ROOT / 'tools/reference/xsum_perf_screen.py'),
                               '--upstream', str(ROOT / '.reference/upstream-ncm'), '--source', str(SOURCE),
                               '--inputs', str(INPUTS), '--batch', str(batch), '--no-stage-sync', '--collect-garbage']
                env = os.environ.copy(); env['OMP_NUM_THREADS'] = '1'; env['TOKENIZERS_PARALLELISM'] = 'false'
                if runtime == 'rust': env['LD_LIBRARY_PATH'] = str(cudnn)
                with (out / f'{name}.stdout').open('xb') as stdout, (out / f'{name}.stderr').open('xb') as stderr:
                    process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(out / f'{name}.time'), *command],
                                               cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                    started = time.monotonic()
                    try:
                        code = process.wait(timeout=900)
                        status = 'complete' if code == 0 else 'failed'
                    except subprocess.TimeoutExpired:
                        terminate_group(process); code = process.returncode; status = 'timeout'
                    result = {'name': name, 'status': status, 'returncode': code,
                              'pid': process.pid, 'start_monotonic': started, 'end_monotonic': time.monotonic()}
                    process = None
                if status == 'complete':
                    value = json.loads((out / f'{name}.stdout').read_text().splitlines()[-1])
                    if value.get('batch') != batch or value.get('warmups') != 2 or len(value.get('trials_seconds', [])) != 5:
                        raise ValueError(f'{name}: incomplete result')
                    if runtime == 'rust' and value.get('backend') != 'strict-cudnn':
                        raise ValueError(f'{name}: strict backend required')
                    if runtime == 'python' and len(value.get('cleanup_seconds', [])) != 5:
                        raise ValueError(f'{name}: missing cleanup timings')
                    codes = value['codes'] if runtime == 'rust' else [x for row in value['codes'] for x in row]
                    ids = value['ids'] if runtime == 'rust' else [x for row in value['ids'] for x in row]
                    if len(codes) != batch * 200 or len(ids) != batch * 48: raise ValueError('wrong output dimensions')
                    if batch in signatures:
                        result['outputs_match_other_runtime'] = signatures[batch] == (codes, ids)
                    else:
                        signatures[batch] = (codes, ids)
                    (out / f'{name}.json').write_text(json.dumps(value, sort_keys=True) + '\n')
                results.append(result)
                (out / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(result), flush=True)
    finally:
        if process is not None: terminate_group(process)
        stop.set(); thread.join(timeout=15)
        if thread.is_alive(): raise RuntimeError('activity monitor did not stop')
    if any(digest(p) != source_hashes[str(p.relative_to(ROOT))] for p in paths): raise ValueError('source changed')
    if any(digest(p) != cudnn_hashes[p.name] for p in cudnn_files): raise ValueError('cuDNN changed')
    (out / 'collection.json').write_text(json.dumps({'jobs': results}, indent=2) + '\n')


if __name__ == '__main__': main()
