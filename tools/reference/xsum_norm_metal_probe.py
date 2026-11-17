"""Compare a bounded Metal Welford normalization with captured CUDA output."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

HEADER = r'''
struct Stats { float mean; float sigma; float count; };
inline Stats online(float v, Stats s) {
    float d = v - s.mean; float n = s.count + 1.0f;
    float m = s.mean + d * (1.0f / n);
    return {m, s.sigma + d * (v - m), n};
}
inline Stats combine(Stats b, Stats a) {
    float d = b.mean - a.mean; float n = a.count + b.count;
    if (n <= 0.0f) return {0.0f, 0.0f, 0.0f};
    float c = 1.0f / n; float na = a.count * c; float nb = b.count * c;
    return {na * a.mean + nb * b.mean,
            a.sigma + b.sigma + d * d * a.count * nb, n};
}
'''
SOURCE = r'''
uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32u;
uint warp = tid / 32u;
uint row = threadgroup_position_in_grid.x;
if (threads_per_simdgroup != 32u) {
    for (uint col = tid; col < 768u; col += 128u) out[row * 768u + col] = NAN;
    return;
}
threadgroup float mean_buf[4];
threadgroup float sigma_buf[4];
threadgroup float count_buf[4];
Stats s{0.0f, 0.0f, 0.0f};
for (uint v = tid; v < 192u; v += 128u) {
    for (uint j = 0; j < 4u; ++j) s = online(x[row * 768u + 4u * v + j], s);
}
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
if (tid == 0u) { mean_buf[0] = s.mean; sigma_buf[0] = s.sigma / 768.0f; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = rsqrt(sigma_buf[0] + 1.0e-6f);
for (uint v = tid; v < 192u; v += 128u) {
    for (uint j = 0; j < 4u; ++j) {
        uint col = 4u * v + j;
        out[row * 768u + col] = gamma[col] * (inv * (x[row * 768u + col] - mean_buf[0])) + beta[col];
    }
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--math-mode', choices=['safe', 'relaxed', 'fast'], default='safe')
    args = p.parse_args()
    if args.output.exists() or sha(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    data = mx.load(str(args.input))
    if set(data) != {'x', 'gamma', 'beta', 'cuda'} or data['x'].shape != (512, 768) or data['cuda'].shape != (512, 768) or data['gamma'].shape != (768,) or data['beta'].shape != (768,):
        raise ValueError('unexpected input shape')
    x, gamma, beta = data['x'], data['gamma'], data['beta']
    kernel = mx.fast.metal_kernel(name='ncm_xsum_welford_probe', input_names=['x', 'gamma', 'beta'],
                                 output_names=['out'], source=SOURCE, header=HEADER,
                                 compile_options={'math_mode': args.math_mode})
    (candidate,) = kernel(inputs=[x, gamma, beta], grid=(512 * 128, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(512, 768)], output_dtypes=[mx.float32])
    baseline = mx.fast.layer_norm(x, gamma, beta, 1e-6)
    mx.eval(candidate, baseline)
    expected = np.asarray(data['cuda']); result = {}
    for name, value in [('mlx_fast', baseline), ('welford', candidate)]:
        actual = np.asarray(value); diff = np.abs(actual - expected)
        if not np.isfinite(actual).all():
            raise ValueError('nonfinite result')
        result[name] = {'exact_elements': int(np.count_nonzero(diff == 0)), 'total_elements': int(diff.size),
                        'max_abs': float(diff.max()), 'mean_abs': float(diff.mean())}
    if sha(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256, 'math_mode': args.math_mode, 'scope': '512 identical prefix rows; non-production diagnostic', 'results': result}, indent=2) + '\n')


if __name__ == '__main__':
    main()
