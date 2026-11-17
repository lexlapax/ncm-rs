"""Compare controlled XSum head projection chains on one captured row."""
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
    shapes = {'hidden': (1, 6, 768), 'weight': (384, 768), 'book': (4096, 384),
              'norm': (4096,), 'scales': (2,), 'cuda': (1, 6, 4096)}
    if set(d) != set(shapes) or any(d[k].shape != shape for k, shape in shapes.items()):
        raise ValueError('unexpected captured shape')
    h = mx.concatenate([mx.zeros((9, 6, 768)), d['hidden'], mx.zeros((22, 6, 768))], axis=0)
    flat = mx.reshape(h, (192, 768))
    w = mx.transpose(d['weight'])
    book = mx.transpose(d['book'])
    results = {}
    expected = np.asarray(d['cuda'])
    for parts in (1, 2, 3, 4, 6):
        step = 768 // parts
        products = [flat[:, i:i + step] @ w[i:i + step] for i in range(0, 768, step)]
        projected = products[0]
        for item in products[1:]:
            projected = projected + item
        dot = projected[:, :192] @ book[:192] + projected[:, 192:] @ book[192:]
        score = mx.reshape(((dot * 2.0 - d['norm']) * d['scales'][0]) * d['scales'][1],
                           (32, 6, 4096))[9:10]
        mx.eval(score)
        actual = np.asarray(score)
        if not np.isfinite(actual).all():
            raise ValueError('nonfinite score')
        diff = np.abs(actual - expected)
        results[str(parts)] = {'exact_elements': int(np.count_nonzero(diff == 0)),
            'total_elements': int(diff.size), 'max_abs': float(diff.max()),
            'pair_margin': float(actual[0, 4, 3182] - actual[0, 4, 3690]),
            'choice': int(actual[0, 4].argmax())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'input_sha256': args.input_sha256,
        'scope': 'one supplied hidden row, controlled CUDA norms/scales and original batch geometry',
        'cuda_pair_margin': float(expected[0, 4, 3182] - expected[0, 4, 3690]),
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
