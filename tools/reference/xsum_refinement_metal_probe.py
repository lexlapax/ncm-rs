"""Compare XSum refinement on an identical captured CUDA/Metal input."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    data = mx.load(str(args.input))
    shapes = {'input': (1, 48, 384), 'weight': (384, 384, 3), 'bias': (384,),
              'cuda_conv': (1, 48, 384), 'cuda_reconstruction': (1, 48, 384),
              'metal_reconstruction': (1, 48, 384)}
    if set(data) != set(shapes) or any(data[k].shape != v for k, v in shapes.items()):
        raise ValueError('unexpected captured tensor shape')
    x = mx.concatenate([mx.zeros((9, 48, 384)), data['input'],
                        mx.zeros((22, 48, 384))], axis=0)
    convolution = mx.conv1d(x, mx.transpose(data['weight'], (0, 2, 1)), padding=1) + data['bias']
    reconstruction = x * 0.5 + convolution * 0.5
    mx.eval(convolution, reconstruction)
    results = {}
    for name, actual, expected in (
        ('convolution', convolution[9:10], data['cuda_conv']),
        ('reconstruction', reconstruction[9:10], data['cuda_reconstruction']),
        ('captured_metal_reconstruction', reconstruction[9:10], data['metal_reconstruction']),
    ):
        a, b = np.asarray(actual), np.asarray(expected)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite comparison')
        difference = np.abs(a - b)
        results[name] = {'different': int(np.count_nonzero(difference)),
                         'total': int(difference.size), 'max_abs': float(difference.max())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
                                      'scope': 'one identical captured row in batch 32',
                                      'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
