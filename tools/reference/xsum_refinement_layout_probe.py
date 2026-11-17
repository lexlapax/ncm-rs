"""Screen explicit Metal convolution layouts against one CUDA refinement input."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tensors', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.tensors.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    d = mx.load(str(args.input))
    shapes = {'input': (1, 48, 384), 'weight': (384, 384, 3), 'bias': (384,),
              'cuda_conv': (1, 48, 384), 'cuda_reconstruction': (1, 48, 384),
              'metal_reconstruction': (1, 48, 384)}
    if set(d) != set(shapes) or any(d[k].shape != v for k, v in shapes.items()):
        raise ValueError('unexpected tensor shape')
    x = mx.concatenate([mx.zeros((9, 48, 384)), d['input'],
                        mx.zeros((22, 48, 384))], axis=0)
    padded = mx.concatenate([mx.zeros((32, 1, 384)), x, mx.zeros((32, 1, 384))], axis=1)
    windows = [padded[:, i:i + 48, :] for i in range(3)]
    products = [windows[i] @ mx.transpose(d['weight'][:, :, i]) for i in range(3)]
    # The flattened layout follows the original weight's channel, kernel order.
    patches = mx.stack(windows, axis=-1).reshape((1536, 1152))
    weight = d['weight'].reshape((384, 1152))
    candidates = {
        'flattened': (patches @ mx.transpose(weight)).reshape((32, 48, 384)) + d['bias'],
        'three_forward': ((products[0] + products[1]) + products[2]) + d['bias'],
        'three_reverse': ((products[2] + products[1]) + products[0]) + d['bias'],
    }
    results, tensors = {}, {}
    for name, conv in candidates.items():
        recon = x * 0.5 + conv * 0.5
        mx.eval(conv, recon)
        output = recon[9:10]
        tensors[name] = output
        results[name] = {}
        for label, actual, reference in [('convolution', conv[9:10], d['cuda_conv']),
                                         ('reconstruction', output, d['cuda_reconstruction'])]:
            a, b = np.asarray(actual), np.asarray(reference)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite comparison')
            diff = np.abs(a - b)
            results[name][label] = {'different': int(np.count_nonzero(diff)),
                                    'total': int(diff.size), 'max_abs': float(diff.max())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    mx.save_safetensors(str(args.tensors), tensors)
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'tensors_sha256': digest(args.tensors), 'scope': 'one captured input in batch 32',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
