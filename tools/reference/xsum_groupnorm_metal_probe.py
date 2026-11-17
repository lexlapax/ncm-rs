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

REFERENCE_HEADER = r'''
struct GroupStats { float mean; float m2; float count; };
inline GroupStats group_add(GroupStats a, float x) {
    float n = a.count + 1.0f, delta = x - a.mean;
    float mean = a.mean + delta / n;
    return {mean, a.m2 + delta * (x - mean), n};
}
inline GroupStats group_combine(GroupStats a, GroupStats b) {
    if (a.count == 0.0f) return b;
    if (b.count == 0.0f) return a;
    float delta = b.mean - a.mean, n = a.count + b.count, ratio = b.count / n;
    return {a.mean + delta * ratio, a.m2 + b.m2 + delta * delta * a.count * ratio, n};
}
inline GroupStats group_reduce(GroupStats a) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        GroupStats b{simd_shuffle_down(a.mean, offset), simd_shuffle_down(a.m2, offset),
                     simd_shuffle_down(a.count, offset)};
        a = group_combine(a, b);
    }
    return a;
}
inline float refined_rsqrt(float v) {
    float y = rsqrt(v);
    float square = y * y;
    float square_error = fma(y, y, -square);
    float residual = fma(-v, square, 1.0f) - v * square_error;
    return fma(y, 0.5f * residual, y);
}
'''
REFERENCE_SOURCE = r'''
uint tid = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.x;
if (threads_per_simdgroup != 32u) {
    if (tid == 0u) { stats[row * 3u] = NAN; stats[row * 3u + 1u] = NAN; stats[row * 3u + 2u] = NAN; }
    for (uint col = tid; col < 576u; col += 512u) out[row * 576u + col] = NAN;
    return;
}
GroupStats value{0.0f, 0.0f, 0.0f};
for (uint j = tid; j < 144u; j += 512u)
    for (uint k = 0; k < 4u; ++k) value = group_add(value, x[row * 576u + j * 4u + k]);
value = group_reduce(value);
threadgroup float means[16], m2s[16], counts[16];
if (tid % 32u == 0u) { means[tid / 32u] = value.mean; m2s[tid / 32u] = value.m2; counts[tid / 32u] = value.count; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 32u) {
    value = tid < 16u ? GroupStats{means[tid], m2s[tid], counts[tid]} : GroupStats{0.0f, 0.0f, 0.0f};
    value = group_reduce(value);
}
if (tid == 0u) {
    means[0] = value.mean; m2s[0] = rsqrt(value.m2 / value.count + 1.0e-5f);
    stats[row * 3u] = means[0]; stats[row * 3u + 1u] = m2s[0];
    stats[row * 3u + 2u] = value.m2 / value.count + 1.0e-5f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint col = tid; col < 576u; col += 512u)
    out[row * 576u + col] = (x[row * 576u + col] - means[0]) * m2s[0];
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-math-mode', choices=['safe', 'relaxed', 'fast'], default='safe')
    parser.add_argument('--reference-inverse', choices=['rsqrt', 'sqrt-div', 'refined'], default='rsqrt')
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
    source = REFERENCE_SOURCE
    if args.reference_inverse == 'sqrt-div':
        source = source.replace('rsqrt(value.m2 / value.count + 1.0e-5f)',
                                '(1.0f / sqrt(value.m2 / value.count + 1.0e-5f))')
    elif args.reference_inverse == 'refined':
        source = source.replace('rsqrt(value.m2 / value.count + 1.0e-5f)',
                                'refined_rsqrt(value.m2 / value.count + 1.0e-5f)')
    reference_kernel = mx.fast.metal_kernel(name='ncm_xsum_groupnorm_reference_order_probe',
        input_names=['x'], output_names=['out', 'stats'], source=source,
        header=REFERENCE_HEADER, compile_options={'math_mode': args.reference_math_mode})
    reference_order, stats = reference_kernel(inputs=[x], grid=(8 * 512, 1, 1),
        threadgroup=(512, 1, 1), output_shapes=[(8, 576), (8, 3)],
        output_dtypes=[mx.float32, mx.float32])
    mx.eval(baseline, fast, candidate, reference_order, stats)
    stats_values = np.asarray(stats)
    if not np.isfinite(stats_values).all():
        raise ValueError('nonfinite statistics')
    expected = np.asarray(data['cuda'])
    results = {}
    for name, value in [('composed', baseline), ('mlx_fast', fast), ('welford', candidate),
                        ('reference_order', reference_order)]:
        actual = np.asarray(value)
        if not np.isfinite(actual).all():
            raise ValueError('nonfinite output')
        diff = np.abs(actual - expected)
        results[name] = {'exact_elements': int(np.count_nonzero(diff == 0)),
                         'total_elements': int(diff.size), 'max_abs': float(diff.max()),
                         'mean_abs': float(diff.mean()),
                         'different_per_group': [int(np.count_nonzero(row)) for row in diff]}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'reference_math_mode': args.reference_math_mode,
        'reference_inverse': args.reference_inverse,
        'reference_group_mean_rstd_variance_epsilon': stats_values.tolist(),
        'scope': 'one previously inspected forced-history row; non-production diagnostic',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
