"""Screen same-input XSum target projection layouts on Metal."""
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
    shapes = {'input': (1, 6, 384), 'weight': (768, 384), 'bias': (768,),
              'level': (768,), 'cuda_projection': (1, 6, 768), 'metal_projection': (1, 6, 768)}
    if set(d) != set(shapes) or any(d[k].shape != v for k, v in shapes.items()):
        raise ValueError('unexpected tensor shape')
    x = mx.concatenate([mx.zeros((9, 6, 384)), d['input'], mx.zeros((22, 6, 384))]).reshape(192, 384)
    weight = mx.transpose(d['weight'])
    candidates = {'addmm': mx.addmm(d['bias'], x, weight), 'matmul_add': x @ weight + d['bias']}
    for parts in (2, 3, 4, 6):
        width = 384 // parts
        products = [x[:, i:i + width] @ weight[i:i + width] for i in range(0, 384, width)]
        y = products[0]
        for part in products[1:]:
            y = y + part
        candidates[f'split_{parts}'] = y + d['bias']
    results, tensors = {}, {}
    for name, value in candidates.items():
        actual = value.reshape(32, 6, 768)[9:10]
        mx.eval(actual)
        tensors[name] = actual
        a = np.asarray(actual)
        results[name] = {}
        for label in ('cuda_projection', 'metal_projection'):
            b = np.asarray(d[label])
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite comparison')
            diff = np.abs(a - b)
            results[name][label] = {'different': int(np.count_nonzero(diff)),
                                    'total': int(diff.size), 'max_abs': float(diff.max())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    mx.save_safetensors(str(args.tensors), tensors)
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'tensors_sha256': digest(args.tensors), 'scope': 'one captured latent in batch 32',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
