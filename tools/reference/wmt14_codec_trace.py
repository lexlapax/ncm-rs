"""Capture the pinned WMT14 upstream codec trace on real CUDA, without patches."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from codec_trace import POLICY, capture, category
from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, tensor_bytes, verify_input
from wmt14_inventory import INPUTS

REFERENCE_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'


def codec_inputs(rows):
    if len(rows) != 3003 or any(type(row) is not list or any(type(x) is not int or not 0 <= x < 151680 for x in row)
                                for row in rows[:2]):
        raise ValueError('invalid reference rows')
    return [(row[:64] + [151643] * 64)[:64] for row in rows[:2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_VISIBLE_DEVICES',
                                      'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('unsupported environment override')
    if torch.__version__.split('+')[0] != '2.15.0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('pinned Torch and exactly one CUDA device required')
    upstream = args.upstream.resolve()
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != UPSTREAM or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if args.reference.is_symlink() or digest_file(args.reference) != REFERENCE_SHA:
        raise ValueError('tokenizer reference differs')
    rows = json.loads(args.reference.read_bytes())['rows']
    inputs = codec_inputs(rows)
    watched = [Path(__file__).resolve(), Path(__file__).with_name('codec_trace.py').resolve(),
               Path(__file__).with_name('sudoku_artifact.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(),
               Path(__file__).with_name('wmt14_inventory.py').resolve(), args.reference.resolve()]
    watched += sorted((upstream / 'models').glob('*.py'))
    before = {str(p): digest_file(p) for p in watched}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'],
                                  text=True).strip().splitlines()
    if len(gpu) != 1:
        raise ValueError('GPU identity is ambiguous')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    if config.task != 'wmt14':
        raise ValueError('unexpected task')
    state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**config.tokenizer.vqvae)
    codec.load_state_dict(state['model'], strict=True)
    codec = codec.cuda().eval()
    out.mkdir(parents=True)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        traces, decisions = capture(codec, torch.tensor(inputs, dtype=torch.int64, device='cuda'))
    save_file(traces, out / 'traces.safetensors')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if (before != {str(p): digest_file(p) for p in watched} or
            subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != head or
            subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('source changed during capture')
    report = {'schema_version': 1, 'scope': 'two WMT14 source-derived codec inputs; no Rust or quality claim',
              'upstream': head, 'source_sha256': before, 'input_sha256': REFERENCE_SHA,
              'checkpoint_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
              'gpu_identity_and_driver': gpu[0], 'gpu_name': torch.cuda.get_device_name(0),
              'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'python': platform.python_version(),
              'tf32': False, 'sdpa': 'math', 'policy': POLICY, 'reference_patches': [],
              'inputs': inputs, 'trace_sha256': digest_file(out / 'traces.safetensors'), 'decisions': decisions,
              'traces': {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'policy': category(name),
                                'sha256': hashlib.sha256(tensor_bytes(tensor)).hexdigest()}
                         for name, tensor in traces.items()}}
    json_file(out / 'report.json', report)
    print(json.dumps({'tensors': len(traces), 'trace_sha256': report['trace_sha256'],
                      'report_sha256': digest_file(out / 'report.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
