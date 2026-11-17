"""Compare captured CUDA GELU values on identical input with Metal."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gelu(x):
    return (x * mx.array(0.5, mx.float32)) * (
        mx.erf(x / mx.array(np.float32(np.sqrt(2.0)), mx.float32)) + mx.array(1.0, mx.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    data = mx.load(str(args.input))
    if set(data) != {'x', 'cuda'} or data['x'].shape != (1, 518, 3072) or data['cuda'].shape != data['x'].shape:
        raise ValueError('unexpected captured shape')
    x = data['x']
    composed = gelu(x)
    compiled = mx.compile(gelu)(x)
    mx.eval(composed, compiled)
    expected = np.asarray(data['cuda'])
    result = {}
    for name, value in [('composed', composed), ('compiled', compiled)]:
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite output')
        diff = np.abs(value - expected)
        result[name] = {'exact_elements': int(np.count_nonzero(diff == 0)),
                        'total_elements': int(diff.size), 'max_abs': float(diff.max()),
                        'mean_abs': float(diff.mean()),
                        'target4_max_abs': float(diff[0, 516].max())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'scope': 'one captured forced-history row, identical input; diagnostic only',
        'results': result}, indent=2) + '\n')


if __name__ == '__main__':
    main()
