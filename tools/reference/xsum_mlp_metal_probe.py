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
    args = p.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    d = mx.load(str(args.input))
    if (set(d) != {'x', 'weight', 'bias', 'cuda'} or d['x'].shape != (1, 518, 768)
            or d['weight'].shape != (3072, 768) or d['bias'].shape != (3072,)
            or d['cuda'].shape != (1, 518, 3072)):
        raise ValueError('unexpected captured shape')
    # Row 9 occupies its original position; other rows are zero-filled.
    x = mx.concatenate([mx.zeros((9, 518, 768)), d['x'], mx.zeros((22, 518, 768))], axis=0)
    output = (x @ mx.transpose(d['weight']) + d['bias'])[9:10]
    mx.eval(output)
    actual, expected = np.asarray(output), np.asarray(d['cuda'])
    if not np.isfinite(actual).all():
        raise ValueError('nonfinite output')
    delta = np.abs(actual - expected)
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed')
    result = {'input_sha256': args.input_sha256,
        'scope': 'one captured row in zero-filled original batch geometry; diagnostic only',
        'exact_elements': int(np.count_nonzero(delta == 0)), 'total_elements': int(delta.size),
        'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
        'target4_max_abs': float(delta[0, 516].max())}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
