"""Prepare a bounded high-precision mathematical GELU intervention.

This is diagnostic data, not a model artifact or a replacement oracle.
"""
import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    tensors = load_file(args.input)
    if set(tensors) != {'x', 'cuda'} or any(t.shape != (1, 518, 3072) or t.dtype != torch.float32 for t in tensors.values()):
        raise ValueError('unexpected input shape or type')
    if not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('nonfinite input')
    torch.set_num_threads(1)
    values = tensors['x']
    scaled = (values * torch.tensor(0.7071067811865476, dtype=torch.float32)).float()
    erf = torch.erf(scaled.double()).float()
    output = ((values * 0.5).float() * (erf + 1.0).float()).float().contiguous()
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    save_file({'gelu': output}, args.output)
    print(digest(args.output))


if __name__ == '__main__':
    main()
