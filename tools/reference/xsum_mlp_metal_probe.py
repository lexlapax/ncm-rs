"""Compare the captured XSum MLP projection at its original batch geometry."""
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
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--projection', choices=['mlp', 'head'], default='mlp')
    args = p.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    d = mx.load(str(args.input))
    length, width = (518, 3072) if args.projection == 'mlp' else (6, 384)
    if (set(d) != {'x', 'weight', 'bias', 'cuda'} or d['x'].shape != (1, length, 768)
            or d['weight'].shape != (width, 768) or d['bias'].shape != (width,)
            or d['cuda'].shape != (1, length, width)):
        raise ValueError('unexpected captured shape')
    # Row 9 occupies its original position; other rows are zero-filled.
    x = mx.concatenate([mx.zeros((9, length, 768)), d['x'], mx.zeros((22, length, 768))], axis=0)
    projected = x @ mx.transpose(d['weight'])
    output = (projected + d['bias'] if args.projection == 'mlp' else projected)[9:10]
    flat = mx.reshape(x, (32 * length, 768))
    book = mx.transpose(d['weight'])
    split = mx.reshape(flat[:, :384] @ book[:384] + flat[:, 384:] @ book[384:],
                       (32, length, width))
    split = (split + d['bias'] if args.projection == 'mlp' else split)[9:10]
    alternatives = []
    for parts in (3, 4, 6):
        step = 768 // parts
        products = [flat[:, i:i + step] @ book[i:i + step] for i in range(0, 768, step)]
        combined = products[0]
        for product in products[1:]:
            combined = combined + product
        combined = mx.reshape(combined, (32, length, width))
        alternatives.append((str(parts), (combined + d['bias'] if args.projection == 'mlp' else combined)[9:10]))
    mx.eval(output, split, *(value for _, value in alternatives))
    actual, expected = np.asarray(output), np.asarray(d['cuda'])
    if not np.isfinite(actual).all():
        raise ValueError('nonfinite output')
    delta = np.abs(actual - expected)
    split_actual = np.asarray(split)
    if not np.isfinite(split_actual).all():
        raise ValueError('nonfinite split output')
    split_delta = np.abs(split_actual - expected)
    alternate_results = {}
    for name, value in alternatives:
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite alternative output')
        difference = np.abs(value - expected)
        alternate_results[name] = {'exact_elements': int(np.count_nonzero(difference == 0)),
                                   'max_abs': float(difference.max()), 'mean_abs': float(difference.mean())}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    result = {'input_sha256': args.input_sha256,
        'scope': 'one captured row in zero-filled original batch geometry; diagnostic only',
        'projection': args.projection,
        'exact_elements': int(np.count_nonzero(delta == 0)), 'total_elements': int(delta.size),
        'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
        'target4_max_abs': float(delta[0, 516 if args.projection == 'mlp' else 4].max()),
        'split2': {'exact_elements': int(np.count_nonzero(split_delta == 0)),
                   'max_abs': float(split_delta.max()), 'mean_abs': float(split_delta.mean()),
                   'target4_max_abs': float(split_delta[0, 516 if args.projection == 'mlp' else 4].max())},
        'other_splits': alternate_results}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
