"""Compare first-block normalization statistics on one fixed XSum input."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from xsum_norm_metal_probe import HEADER, SOURCE


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
    shapes = {'x': (518, 768), 'gamma': (768,), 'beta': (768,), 'cuda': (518, 768),
              'metal': (518, 768), 'mean': (518, 1), 'rstd': (518, 1), 'variance': (518, 1)}
    if set(d) != set(shapes) or any(d[k].shape != v for k, v in shapes.items()):
        raise ValueError('unexpected input shape')
    marker = 'float inv = rsqrt(sigma_buf[0] + 1.0e-6f);'
    source = SOURCE.replace(marker, marker + '\nif (tid == 0u) { stats[row * 3u] = mean_buf[0]; stats[row * 3u + 1u] = sigma_buf[0]; stats[row * 3u + 2u] = inv; }')
    if source == SOURCE:
        raise ValueError('statistics insertion failed')
    results = {}
    for mode in ('safe', 'relaxed', 'fast'):
        kernel = mx.fast.metal_kernel(name=f'ncm_xsum_norm_stats_{mode}',
            input_names=['x', 'gamma', 'beta'], output_names=['out', 'stats'],
            source=source, header=HEADER, compile_options={'math_mode': mode})
        out, stats = kernel(inputs=[d['x'], d['gamma'], d['beta']],
            grid=(518 * 128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(518, 768), (518, 3)], output_dtypes=[mx.float32, mx.float32])
        mx.eval(out, stats)
        arrays = [('output', np.asarray(out), np.asarray(d['cuda'])),
                  ('mean', np.asarray(stats)[:, 0], np.asarray(d['mean'])[:, 0]),
                  ('variance', np.asarray(stats)[:, 1], np.asarray(d['variance'])[:, 0]),
                  ('rstd', np.asarray(stats)[:, 2], np.asarray(d['rstd'])[:, 0])]
        results[mode] = {}
        for name, a, b in arrays:
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite comparison')
            diff = np.abs(a - b)
            results[mode][name] = {'different': int(np.count_nonzero(diff)),
                                   'total': int(diff.size), 'max_abs': float(diff.max())}
    forced_source = SOURCE.replace(marker, 'float inv = rstd[row];').replace(
        'x[row * 768u + col] - mean_buf[0]', 'x[row * 768u + col] - mean[row]')
    if forced_source == SOURCE:
        raise ValueError('forced statistics insertion failed')
    forced_kernel = mx.fast.metal_kernel(name='ncm_xsum_norm_forced_stats',
        input_names=['x', 'gamma', 'beta', 'mean', 'rstd'], output_names=['out'],
        source=forced_source, header=HEADER, compile_options={'math_mode': 'safe'})
    (forced,) = forced_kernel(inputs=[d['x'], d['gamma'], d['beta'], d['mean'], d['rstd']],
        grid=(518 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(518, 768)], output_dtypes=[mx.float32])
    a, b = np.asarray(forced), np.asarray(d['cuda'])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite comparison')
    diff = np.abs(a - b)
    results['forced_cuda_statistics'] = {'different': int(np.count_nonzero(diff)),
        'total': int(diff.size), 'max_abs': float(diff.max())}
    inverse_source = r"""
    uint i = thread_position_in_grid.x;
    if (i >= 518u) return;
    float v = variance[i] + 1.0e-6f;
    float y = rsqrt(v);
    float square = y * y;
    float error = fma(y, y, -square);
    float residual = fma(-v, square, 1.0f) - v * error;
    plain[i] = y;
    sqrt_div[i] = 1.0f / sqrt(v);
    refined[i] = fma(y, 0.5f * residual, y);
    """
    inverse_kernel = mx.fast.metal_kernel(name='ncm_xsum_norm_inverse',
        input_names=['variance'], output_names=['plain', 'sqrt_div', 'refined'],
        source=inverse_source, compile_options={'math_mode': 'safe'})
    outs = inverse_kernel(inputs=[d['variance']], grid=(518, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(518,), (518,), (518,)], output_dtypes=[mx.float32] * 3)
    expected = np.asarray(d['rstd']).reshape(518)
    results['inverse_from_cuda_variance'] = {}
    for name, out in zip(('plain', 'sqrt_div', 'refined'), outs):
        a = np.asarray(out)
        if not np.isfinite(a).all():
            raise ValueError('nonfinite inverse')
        diff = np.abs(a - expected)
        results['inverse_from_cuda_variance'][name] = {'different': int(np.count_nonzero(diff)),
            'total': int(diff.size), 'max_abs': float(diff.max())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'scope': 'one candidate input, 518 normalization vectors', 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
