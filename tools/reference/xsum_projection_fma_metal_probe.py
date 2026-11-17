"""Screen CUDA-like FMA accumulation orders for XSum projection on Metal."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(lanes, layout, bias_first):
    init = ', '.join('bias[col]' if bias_first and i == 0 else '0.0f' for i in range(lanes))
    index = f'(i * {lanes}u + j)' if layout == 'interleaved' else f'(j * {384 // lanes}u + i)'
    bias = '' if bias_first else 'value += bias[col];'
    return f'''
    uint id = thread_position_in_grid.x;
    if (id >= 23040u) return;
    uint row = id / 768u, col = id % 768u;
    float accum[{lanes}] = {{{init}}};
    for (uint i = 0; i < {384 // lanes}u; ++i) {{
        for (uint j = 0; j < {lanes}u; ++j) {{
            uint k = {index};
            accum[j] = fma(x[row * 384u + k], w[col * 384u + k], accum[j]);
        }}
    }}
    float value = accum[0];
    for (uint j = 1; j < {lanes}u; ++j) value += accum[j];
    {bias}
    out[id] = value;
    '''


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
    results = {}
    for lanes in (1, 2, 4, 8, 16):
        for layout in ('interleaved', 'contiguous'):
            for bias_first in (False, True):
                name = f'{lanes}_{layout}_{"first" if bias_first else "last"}'
                kernel = mx.fast.metal_kernel(name=f'ncm_xsum_fma_{name}',
                    input_names=['x', 'w', 'bias'], output_names=['out'],
                    source=source(lanes, layout, bias_first), compile_options={'math_mode': 'safe'})
                (out,) = kernel(inputs=[data['input'], data['weight'], data['bias']],
                    grid=(23040, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(5, 6, 768)], output_dtypes=[mx.float32])
                actual = np.asarray(out)
                if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                    raise ValueError('nonfinite comparison')
                diff = np.abs(actual - expected)
                results[name] = [{'different': int(np.count_nonzero(row)),
                                  'total': int(row.size), 'max_abs': float(row.max())}
                                 for row in diff]
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'scope': 'one captured and four seeded synthetic inputs, fixed CUDA projection batch geometry',
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
