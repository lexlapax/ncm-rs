"""Compare one strict-CUDA XSum CLI batch with the pinned Python reference."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from run_phase4_xsum_bench import ARTIFACT, INPUTS, ROOT, digest, terminate_group

REF = ROOT / 'reports/phase4-xsum-cuda-performance-b32-v2/python-b32-r0.json'
REF_SHA = 'b2a20dbad883417406387726d799c8f79ecf814507f58d01bd05204e16775125'
INPUT_SHA = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'
ARTIFACT_SHA = '5df37749e64418c586472a4678600603992357b9bc5e3611da307918e13b237e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cudnn-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'CUDA_INJECTION64_PATH',
                 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')
    if any(os.environ.get(key) for key in forbidden): raise ValueError('unsupported environment override')
    out = args.output.resolve(); binary = args.binary.resolve(); cudnn = args.cudnn_dir.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be a new directory under reports')
    if not binary.is_file() or not (cudnn / 'libcudnn.so.9').is_file(): raise ValueError('missing binary or cuDNN')
    if digest(REF) != REF_SHA or digest(INPUTS) != INPUT_SHA or digest(ARTIFACT / 'manifest.json') != ARTIFACT_SHA:
        raise ValueError('reference, inputs or artifact differ')
    reference = json.loads(REF.read_bytes())
    if reference.get('batch') != 32 or reference.get('backend') != 'pinned Python CUDA':
        raise ValueError('unexpected reference')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1: raise ValueError('exactly one GPU required')
    paths = [binary, ARTIFACT / 'manifest.json', INPUTS, REF, ROOT / 'crates/ncm-cli/src/main.rs',
             Path(__file__).resolve()] + sorted(cudnn.glob('libcudnn*.so.9'))
    hashes = {str(p): digest(p) for p in paths}
    out.mkdir(parents=True)
    command = [str(binary), 'sample-xsum', str(ARTIFACT), 'cuda-cudnn', str(INPUTS)]
    env = os.environ.copy(); env['LD_LIBRARY_PATH'] = str(cudnn); env['OMP_NUM_THREADS'] = '1'
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        try: stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            terminate_group(process); stdout, stderr = process.communicate()
            (out / 'stdout.json').write_bytes(stdout); (out / 'stderr.txt').write_bytes(stderr)
            raise TimeoutError('XSum CLI timed out')
    finally:
        if process.poll() is None: terminate_group(process)
    (out / 'stdout.json').write_bytes(stdout); (out / 'stderr.txt').write_bytes(stderr)
    if process.returncode: raise RuntimeError(f'CLI exited {process.returncode}')
    actual = json.loads(stdout)
    if set(actual) != {'codes', 'ids'} or actual['codes'] != reference['codes'] or actual['ids'] != reference['ids']:
        raise AssertionError('CLI codes or IDs differ from Python')
    if any(digest(p) != hashes[str(p)] for p in paths): raise ValueError('CLI inputs changed during check')
    result = {'command': command, 'gpu_identity_and_driver': gpu[0], 'rows': 32,
              'codes_and_ids_match': True, 'source_sha256': hashes, 'output_sha256': digest(out / 'stdout.json'),
              'scope': 'single correctness comparison; not a latency or memory gate'}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
