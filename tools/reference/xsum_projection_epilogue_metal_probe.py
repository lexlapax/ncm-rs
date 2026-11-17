"""Screen bias epilogues after a CUDA-equivalent XSum projection dot product."""
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
    data = mx.load(str(args.input))
    shapes = {'input': (5, 6, 384), 'weight': (768, 384), 'bias': (768,), 'cuda': (5, 6, 768)}
    if set(data) != set(shapes) or any(data[k].shape != v for k, v in shapes.items()):
        raise ValueError('unexpected input shape')
    expected = np.asarray(data['cuda'])
    expressions = {'sum_then_bias': '(a + b) + bias[col]',
                   'first_then_bias': '(a + bias[col]) + b',
                   'second_then_bias': 'a + (b + bias[col])'}
    results = {}
    for name, expression in expressions.items():
        source = f'''
        uint id = thread_position_in_grid.x;
        if (id >= 23040u) return;
        uint row = id / 768u, col = id % 768u;
        float a = 0.0f, b = 0.0f;
        for (uint i = 0; i < 192u; ++i) {{
            a = fma(x[row * 384u + i], w[col * 384u + i], a);
            b = fma(x[row * 384u + i + 192u], w[col * 384u + i + 192u], b);
        }}
        out[id] = {expression};
        '''
        kernel = mx.fast.metal_kernel(name=f'ncm_xsum_epilogue_{name}',
            input_names=['x', 'w', 'bias'], output_names=['out'], source=source,
            compile_options={'math_mode': 'safe'})
        (out,) = kernel(inputs=[data['input'], data['weight'], data['bias']],
            grid=(23040, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(5, 6, 768)], output_dtypes=[mx.float32])
        actual = np.asarray(out)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('nonfinite comparison')
        diff = np.abs(actual - expected)
        results[name] = [{'different': int(np.count_nonzero(row)),
                          'total': int(row.size), 'max_abs': float(row.max())} for row in diff]
    matmul_results = []
    for i in range(5):
        batch = mx.concatenate([mx.zeros((9, 6, 384)), data['input'][i:i + 1],
                                mx.zeros((22, 6, 384))], axis=0).reshape(192, 384)
        w = mx.transpose(data['weight'])
        first = mx.addmm(data['bias'], batch[:, :192], w[:192, :])
        second = batch[:, 192:] @ w[192:, :]
        actual = np.asarray((first + second).reshape(32, 6, 768)[9])
        diff = np.abs(actual - expected[i])
        matmul_results.append({'different': int(np.count_nonzero(diff)),
                               'total': int(diff.size), 'max_abs': float(diff.max())})
    results['matmul_first_then_bias'] = matmul_results
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'scope': 'one captured and four seeded synthetic inputs', 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
