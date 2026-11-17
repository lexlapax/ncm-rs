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


def gelu_multiply(x):
    return (x * mx.array(0.5, mx.float32)) * (
        mx.erf(x * mx.array(np.float32(1.0 / np.sqrt(2.0)), mx.float32)) + mx.array(1.0, mx.float32))


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
    multiply = gelu_multiply(x)
    compiled_multiply = mx.compile(gelu_multiply)(x)
    kernel_dir = Path('.reference/mlx-native/include/mlx/backend/metal/kernels')
    erf_header = (kernel_dir / 'erf.h').read_text()
    expm1_header = (kernel_dir / 'expm1f.h').read_text()
    include = '#include "mlx/backend/metal/kernels/expm1f.h"'
    if erf_header.count(include) != 1:
        raise ValueError('unexpected installed MLX erf header')
    header = erf_header.replace(include, expm1_header) + '\n'
    cuda_order = []
    for mode in ('safe', 'relaxed', 'fast'):
        kernel = mx.fast.metal_kernel(name=f'ncm_xsum_gelu_cuda_order_{mode}',
            input_names=['x'], output_names=['out'],
            header=header,
            source='uint i = thread_position_in_grid.x; float v = x[i]; out[i] = v * 0.5f * (1.0f + erf(v * 0.7071067811865475244f));',
            ensure_row_contiguous=True, compile_options={'math_mode': mode})
        (value,) = kernel(inputs=[x], grid=(1591296, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(1, 518, 3072)], output_dtypes=[mx.float32])
        cuda_order.append((f'cuda_order_{mode}', value))
    mx.eval(composed, compiled, multiply, compiled_multiply, *(value for _, value in cuda_order))
    expected = np.asarray(data['cuda'])
    result = {}
    for name, value in [('composed', composed), ('compiled', compiled),
                        ('multiply', multiply), ('compiled_multiply', compiled_multiply), *cuda_order]:
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
        'erf_header_sha256': digest(kernel_dir / 'erf.h'),
        'expm1_header_sha256': digest(kernel_dir / 'expm1f.h'),
        'scope': 'one captured forced-history row, identical input; diagnostic only',
        'results': result}, indent=2) + '\n')


if __name__ == '__main__':
    main()
