"""Compare Metal group-normalization arithmetic on one captured XSum row.

This is an exploratory forced-history probe, not an inference implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from xsum_norm_metal_probe import HEADER

SOURCE = r'''
uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32u;
uint warp = tid / 32u;
uint row = threadgroup_position_in_grid.x;
if (threads_per_simdgroup != 32u) {
    for (uint col = tid; col < 576u; col += 128u) out[row * 576u + col] = NAN;
    return;
}
threadgroup float mean_buf[4];
threadgroup float sigma_buf[4];
threadgroup float count_buf[4];
Stats s{0.0f, 0.0f, 0.0f};
for (uint col = tid; col < 576u; col += 128u) s = online(x[row * 576u + col], s);
for (uint offset = 16u; offset > 0u; offset >>= 1u) {
    Stats other{simd_shuffle_down(s.mean, offset), simd_shuffle_down(s.sigma, offset),
                simd_shuffle_down(s.count, offset)};
    s = combine(s, other);
}
for (uint offset = 2u; offset > 0u; offset >>= 1u) {
    if (lane == 0u && warp >= offset && warp < 2u * offset) {
        uint target = warp - offset; mean_buf[target] = s.mean;
        sigma_buf[target] = s.sigma; count_buf[target] = s.count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u && warp < offset) {
        Stats other{mean_buf[warp], sigma_buf[warp], count_buf[warp]};
        s = combine(s, other);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0u) { mean_buf[0] = s.mean; sigma_buf[0] = s.sigma / 576.0f; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = rsqrt(sigma_buf[0] + 1.0e-5f);
for (uint col = tid; col < 576u; col += 128u)
    out[row * 576u + col] = (x[row * 576u + col] - mean_buf[0]) * inv;
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    data = mx.load(str(args.input))
    if set(data) != {'x', 'cuda'} or data['x'].shape != (8, 576) or data['cuda'].shape != (8, 576):
        raise ValueError('unexpected captured shape')
    x = data['x']
    centered = x - mx.mean(x, axis=-1, keepdims=True)
    baseline = centered * mx.rsqrt(mx.mean(mx.square(centered), axis=-1, keepdims=True) + 1e-5)
    fast = mx.fast.layer_norm(x, None, None, 1e-5)
    kernel = mx.fast.metal_kernel(name='ncm_xsum_groupnorm_probe', input_names=['x'],
                                  output_names=['out'], source=SOURCE, header=HEADER,
                                  compile_options={'math_mode': 'safe'})
    (candidate,) = kernel(inputs=[x], grid=(8 * 128, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(8, 576)], output_dtypes=[mx.float32])
    mx.eval(baseline, fast, candidate)
    expected = np.asarray(data['cuda'])
    results = {}
    for name, value in [('composed', baseline), ('mlx_fast', fast), ('welford', candidate)]:
        actual = np.asarray(value)
        if not np.isfinite(actual).all():
            raise ValueError('nonfinite output')
        diff = np.abs(actual - expected)
        results[name] = {'exact_elements': int(np.count_nonzero(diff == 0)),
                         'total_elements': int(diff.size), 'max_abs': float(diff.max()),
                         'mean_abs': float(diff.mean())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'scope': 'one previously inspected forced-history row; non-production diagnostic',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
