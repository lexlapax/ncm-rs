"""Compare controlled same-input FP32 XSum head arithmetic on Metal."""
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
    args = p.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    d = mx.load(str(args.input))
    if (set(d) != {'z', 'book', 'norm', 'scales', 'cuda', 'cuda_dot'}
            or d['z'].shape != (1, 6, 384) or d['book'].shape != (4096, 384)
            or d['norm'].shape != (4096,) or d['scales'].shape != (2,)
            or d['cuda'].shape != (1, 6, 4096) or d['cuda_dot'].shape != (1, 6, 4096)):
        raise ValueError('unexpected captured shape')
    z = mx.concatenate([mx.zeros((9, 6, 384)), d['z'], mx.zeros((22, 6, 384))], axis=0)
    dot = (z @ mx.transpose(d['book']))[9:10]
    value = ((dot * 2.0 - d['norm']) * d['scales'][0]) * d['scales'][1]
    forced_dot_value = ((d['cuda_dot'] * 2.0 - d['norm']) * d['scales'][0]) * d['scales'][1]
    mx.eval(value, dot, forced_dot_value)
    actual, expected = np.asarray(value), np.asarray(d['cuda'])
    if not np.isfinite(actual).all():
        raise ValueError('nonfinite output')
    diff = np.abs(actual - expected)
    metal_dot = np.asarray(dot)
    expected_dot = np.asarray(d['cuda_dot'])
    forced_dot = np.asarray(forced_dot_value)
    if not np.isfinite(metal_dot).all() or not np.isfinite(forced_dot).all():
        raise ValueError('nonfinite intermediate output')
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    result = {'input_sha256': args.input_sha256,
        'scope': 'one projection row with controlled CUDA norms and scales, original batch geometry',
        'exact_elements': int(np.count_nonzero(diff == 0)), 'total_elements': int(diff.size),
        'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
        'cuda_pair_margin': float(expected[0, 4, 3182] - expected[0, 4, 3690]),
        'metal_pair_margin': float(actual[0, 4, 3182] - actual[0, 4, 3690]),
        'dot_exact_elements': int(np.count_nonzero(metal_dot == expected_dot)),
        'dot_total_elements': int(metal_dot.size),
        'dot_max_abs': float(np.max(np.abs(metal_dot - expected_dot))),
        'cuda_dot_pair': [float(expected_dot[0, 4, i]) for i in (3182, 3690)],
        'metal_dot_pair': [float(metal_dot[0, 4, i]) for i in (3182, 3690)],
        'forced_dot_exact_elements': int(np.count_nonzero(forced_dot == expected)),
        'forced_dot_pair_margin': float(forced_dot[0, 4, 3182] - forced_dot[0, 4, 3690]),
        'cuda_choice': int(expected[0, 4].argmax()), 'metal_choice': int(actual[0, 4].argmax())}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
