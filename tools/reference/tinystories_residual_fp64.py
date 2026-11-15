"""Diagnose TinyStories residual convolution errors against FP64 arithmetic."""
import argparse
import hashlib
import json
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--python-trace', type=Path, required=True)
    p.add_argument('--rust-trace', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    manifest_path = args.artifact / 'manifest.json'
    if sha(manifest_path) != '0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1':
        raise ValueError('artifact manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    codec_path = args.artifact / 'codec.safetensors'
    if sha(codec_path) != manifest['files']['codec.safetensors']['sha256']:
        raise ValueError('codec payload differs')
    python = load_file(args.python_trace, device='cpu')
    rust = load_file(args.rust_trace, device='cpu')
    if python.keys() != rust.keys() or not torch.equal(python['cascade.3'], rust['cascade.3']):
        raise ValueError('residual inputs differ; use separate first-level comparison')
    if python['refined'].shape != rust['refined'].shape or python['cascade.3'].shape != python['refined'].shape:
        raise ValueError('invalid trace shapes')
    weights = load_file(codec_path, device='cpu')
    with torch.inference_mode():
        x = python['cascade.3'].double().cuda().transpose(1, 2).contiguous()
        weight = weights['vq.quant_resi.0.weight'].double().cuda()
        bias = weights['vq.quant_resi.0.bias'].double().cuda()
        ideal = (x * 0.5 + F.conv1d(x, weight, bias, padding=1) * 0.5).transpose(1, 2).cpu()
        pe = (python['refined'].double() - ideal).abs()
        re = (rust['refined'].double() - ideal).abs()
        band = 5e-4 + 2e-4 * python['refined'].double().abs()
        def stat(t):
            return {'max_abs': t.max().item(), 'mean_abs': t.mean().item(),
                    'rms': t.square().mean().sqrt().item(), 'p99': torch.quantile(t.flatten(), .99).item()}
        result = {'elements': ideal.numel(), 'reference_absmax': ideal.abs().max().item(),
                  'python_vs_fp64': stat(pe), 'rust_vs_fp64': stat(re),
                  'rust_closer_count': (re < pe).sum().item(), 'python_closer_count': (pe < re).sum().item(),
                  'equal_count': (pe == re).sum().item(),
                  'python_vs_fp64_outside_phase2_accumulated_band': (pe > band).sum().item(),
                  'rust_vs_fp64_outside_python_error_plus_band': (re > pe + band).sum().item(),
                  'python_trace_sha256': sha(args.python_trace), 'rust_trace_sha256': sha(args.rust_trace),
                  'note': 'diagnostic FP64 comparison; no tolerance is changed and this is not a parity gate'}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
