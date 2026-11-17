"""Collect source-bound WMT14 CUDA pool and PyTorch allocator observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from run_phase4_xsum_bench import digest, monitor, terminate_group

PREFIX = ROOT / 'reports/phase4-wmt14-tokenizer-reference-v1.json'
ARTIFACT = ROOT / 'artifacts/canonical/wmt14-inference-v1-final'
BINARY = ROOT / 'target/release/examples/wmt14_fast_argmax_memory'
PY_PROBE = ROOT / 'tools/reference/wmt14_memory_probe.py'
PYTHON = ROOT / '.reference/venv/bin/python'
CUDNN = ROOT / '.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib'
CELLS = [(0, 1), (0, 4), (0, 16), (0, 32), (1, 1), (1, 4), (1, 16)]


def source_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    return head, dirty


def reference(mode, batch):
    folder = 'b32' if batch == 32 else 'small'
    return ROOT / f'reports/phase4-wmt14-fast-argmax-bench-{folder}-v1/python-cfg{mode}-b{batch}-r0.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be a new directory under reports')
    forbidden = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING',
                 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
    if any(os.environ.get(key) for key in forbidden):
        raise ValueError('unexpected environment override')
    libraries = sorted(CUDNN.glob('libcudnn*.so.9'))
    if not libraries:
        raise ValueError('cuDNN library inventory missing')
    watched = [Path(__file__).resolve(), BINARY, PY_PROBE, PREFIX, ARTIFACT / 'manifest.json',
               ROOT / 'tools/bench/run_phase4_xsum_bench.py', *libraries, *[reference(m, b) for m, b in CELLS]]
    before = {str(path.relative_to(ROOT)): digest(path) for path in watched}
    if (source_state() != (args.head, '') or digest(BINARY) != args.binary_sha256 or
            digest(PREFIX) != '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f' or
            digest(ARTIFACT / 'manifest.json') != '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81'):
        raise ValueError('source, binary, prefixes or artifact identity differs')
    out.mkdir(parents=True)
    manifest = {'head': args.head, 'sha256': before, 'cells': CELLS,
        'scope': 'one process per runtime and cell; three warmups, native allocator snapshots; load separately',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl'), daemon=True)
    thread.start()
    process = None
    try:
        for mode, batch in CELLS:
            expected = json.loads(reference(mode, batch).read_text())['outputs']
            for runtime in (('rust', 'python') if (mode + batch) % 2 == 0 else ('python', 'rust')):
                stem = f'{runtime}-cfg{mode}-b{batch}'
                result_path = out / (stem + '.json')
                if runtime == 'rust':
                    command = [str(BINARY), str(ARTIFACT), str(PREFIX), str(batch), str(mode)]
                else:
                    command = [str(PYTHON), str(PY_PROBE), '--upstream', str(ROOT / '.reference/upstream-ncm'),
                        '--source', str(ROOT / 'artifacts/upstream/wmt14-phase4'), '--prefixes', str(PREFIX),
                        '--output', str(result_path), '--batch', str(batch), '--cfg', str(mode)]
                env = dict(os.environ, OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
                if runtime == 'rust':
                    env['LD_LIBRARY_PATH'] = str(CUDNN)
                with (out / (stem + '.stdout')).open('x') as stdout, (out / (stem + '.stderr')).open('x') as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                    try:
                        code = process.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        terminate_group(process)
                        raise RuntimeError(f'{stem} timed out; evidence retained')
                process = None
                if code:
                    raise RuntimeError(f'{stem} exited {code}; evidence retained')
                if runtime == 'rust':
                    report = json.loads((out / (stem + '.stdout')).read_text().splitlines()[-1])
                    result_path.write_text(json.dumps(report, sort_keys=True) + '\n')
                else:
                    report = json.loads(result_path.read_text())
                if report['batch'] != batch or report['cfg'] != mode or report['outputs'] != expected:
                    raise ValueError(f'{stem} output or workload differs')
                print(stem, 'matched', flush=True)
    finally:
        if process is not None:
            terminate_group(process)
        stop.set()
        thread.join()
    if source_state() != (args.head, '') or before != {str(path.relative_to(ROOT)): digest(path) for path in watched}:
        raise ValueError('source or input changed during observation')
    (out / 'complete.json').write_text(json.dumps({'cells': len(CELLS), 'runtimes_per_cell': 2,
        'results': {path.name: digest(path) for path in sorted(out.glob('*-cfg*-b*.json'))}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
