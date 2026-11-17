"""Compare isolated Metal upsample formulations on one captured XSum batch."""
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
    p.add_argument('--full-reference', type=Path)
    p.add_argument('--full-reference-sha256')
    args = p.parse_args()
    if args.output.exists() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input digest differs')
    if bool(args.full_reference) != bool(args.full_reference_sha256):
        raise ValueError('full reference and digest must be supplied together')
    if args.full_reference and digest(args.full_reference) != args.full_reference_sha256:
        raise ValueError('full reference digest differs')
    d = mx.load(str(args.input))
    if set(d) != {'x', 'weight', 'bias', 'cuda', 'metal'} or d['x'].shape != (32, 4, 384) or d['weight'].shape != (384, 384, 3) or d['bias'].shape != (384,) or d['cuda'].shape != (12, 384) or d['metal'].shape != (12, 384):
        raise ValueError('unexpected probe inputs')
    x, weight, bias = d['x'], d['weight'], d['bias']
    native = mx.conv_transpose1d(x, weight.transpose(1, 2, 0), stride=3) + bias
    flattened = (x @ weight.reshape(384, 1152)).reshape(32, 4, 384, 3).transpose(0, 1, 3, 2).reshape(32, 12, 384) + bias
    separate = mx.stack([x @ weight[:, :, k] for k in range(3)], axis=2).reshape(32, 12, 384) + bias
    mx.eval(native, flattened, separate)
    cuda, prior = np.asarray(d['cuda']), np.asarray(d['metal'])
    full = mx.load(str(args.full_reference)) if args.full_reference else None
    if full is not None and (set(full) != {'cuda_all'} or full['cuda_all'].shape != (32, 12, 384)):
        raise ValueError('unexpected full reference')
    full_expected = np.asarray(full['cuda_all']) if full is not None else None
    if full_expected is not None and not np.array_equal(full_expected[9], cuda):
        raise ValueError('full reference row differs')
    result = {}
    for name, tensor in [('native', native), ('flattened_matmul', flattened), ('separate_matmuls', separate)]:
        all_values = np.asarray(tensor)
        value = all_values[9]
        if not np.isfinite(value).all():
            raise ValueError('nonfinite result')
        diff = np.abs(value - cuda)
        result[name] = {'exact_cuda': int(np.count_nonzero(diff == 0)), 'elements': int(diff.size),
                        'max_abs_cuda': float(diff.max()), 'mean_abs_cuda': float(diff.mean()),
                        'exact_prior_metal': int(np.count_nonzero(value == prior))}
        if full_expected is not None:
            all_diff = np.abs(all_values - full_expected)
            result[name].update({'full_exact_cuda': int(np.count_nonzero(all_diff == 0)),
                                 'full_elements': int(all_diff.size), 'full_max_abs_cuda': float(all_diff.max()),
                                 'full_mean_abs_cuda': float(all_diff.mean())})
    if digest(args.input) != args.input_sha256 or (args.full_reference and digest(args.full_reference) != args.full_reference_sha256):
        raise ValueError('input changed')
    args.output.write_text(json.dumps({'scope': 'isolated batch-32 first upsample, row 4937; non-production diagnostic',
                                       'input_sha256': args.input_sha256, 'full_reference_sha256': args.full_reference_sha256,
                                       'results': result}, indent=2) + '\n')


if __name__ == '__main__':
    main()
