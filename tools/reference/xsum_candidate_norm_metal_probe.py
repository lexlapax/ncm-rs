"""Compare first-block normalizations on the same captured XSum candidate input."""
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
    p.add_argument('--tensors', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.tensors.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    data = mx.load(str(args.input))
    shapes = {'x': (518, 768), 'gamma': (768,), 'beta': (768,),
              'cuda': (518, 768), 'metal': (518, 768)}
    if set(data) != set(shapes) or any(data[k].shape != v for k, v in shapes.items()):
        raise ValueError('unexpected input shape')
    kernel = mx.fast.metal_kernel(name='ncm_xsum_candidate_norm_probe',
        input_names=['x', 'gamma', 'beta'], output_names=['out'], source=SOURCE, header=HEADER,
        compile_options={'math_mode': 'safe'})
    (welford,) = kernel(inputs=[data['x'], data['gamma'], data['beta']],
        grid=(518 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(518, 768)], output_dtypes=[mx.float32])
    fast = mx.fast.layer_norm(data['x'], data['gamma'], data['beta'], 1e-6)
    inverse = 'float inv = rsqrt(sigma_buf[0] + 1.0e-6f);'
    refined = """float v = sigma_buf[0] + 1.0e-6f;
float y = rsqrt(v), square = y * y, error = fma(y, y, -square);
float residual = fma(-v, square, 1.0f) - v * error;
float inv = fma(y, 0.5f * residual, y);"""
    alternate = {}
    for name, replacement in [('refined', refined), ('sqrt_div', 'float inv = 1.0f / sqrt(sigma_buf[0] + 1.0e-6f);')]:
        source = SOURCE.replace(inverse, replacement)
        if source == SOURCE:
            raise ValueError('inverse replacement failed')
        k = mx.fast.metal_kernel(name=f'ncm_xsum_candidate_norm_{name}',
            input_names=['x', 'gamma', 'beta'], output_names=['out'], source=source, header=HEADER,
            compile_options={'math_mode': 'safe'})
        (alternate[name],) = k(inputs=[data['x'], data['gamma'], data['beta']],
            grid=(518 * 128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(518, 768)], output_dtypes=[mx.float32])
    mx.eval(welford, fast, *alternate.values())
    results = {}
    for name, value in [('fast', fast), ('welford', welford), *alternate.items()]:
        results[name] = {}
        a = np.asarray(value)
        for ref in ('cuda', 'metal'):
            b = np.asarray(data[ref])
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite comparison')
            diff = np.abs(a - b)
            results[name][ref] = {'different': int(np.count_nonzero(diff)),
                                  'total': int(diff.size), 'max_abs': float(diff.max())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    mx.save_safetensors(str(args.tensors), {'fast': fast, 'welford': welford, **alternate})
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'tensors_sha256': digest(args.tensors), 'scope': 'one candidate input, 518 vectors',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
