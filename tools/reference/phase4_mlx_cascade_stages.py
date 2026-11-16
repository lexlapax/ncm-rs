"""Localize MLX/CUDA cascade differences using the same captured CUDA inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_codec import prepare_device
from mlx_codec_assets import tensors
from phase4_tinystories_mlx import TinyGenerator


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(actual, expected):
    value = np.asarray(actual)
    if value.shape != expected.shape or not np.isfinite(value).all() or not np.isfinite(expected).all():
        raise ValueError('invalid comparison shape or values')
    ref = expected.astype(np.float64)
    difference = np.abs(value.astype(np.float64) - ref)
    normalized = difference / (5e-4 + 2e-4 * np.abs(ref))
    return {'elements': int(value.size), 'different_bits': int(np.count_nonzero(value.view(np.uint32) != expected.view(np.uint32))),
            'outside_band': int(np.count_nonzero(normalized > 1)), 'max_abs': float(difference.max()),
            'max_normalized': float(normalized.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(args.reference) != args.reference_sha256:
        raise ValueError('output must be new and reference hash must match')
    root = Path(__file__).resolve().parents[2]
    watched = [Path(__file__).resolve(), args.reference.resolve(),
               *(root / 'tools/reference' / name for name in (
                   'phase4_tinystories_mlx.py', 'phase3_mlx_generator.py',
                   'mlx_codec.py', 'mlx_codec_assets.py', 'mlx_operators.py'))]
    watched += sorted(p.resolve() for p in args.artifact.rglob('*') if p.is_file())
    before = {str(p): digest(p) for p in watched}
    prepare_device()
    model = TinyGenerator(args.artifact)
    ref = tensors(args.reference.read_bytes())
    if set(ref) != {'embedding', 'cascade.0', 'cascade.1', 'cascade.2', 'cascade.3',
                    'refined', 'residual.conv', 'residual.conv_half', 'residual.input_half'}:
        raise ValueError('unexpected cascade trace inventory')
    x = mx.array(ref['embedding'])
    results = {}
    for index, stride in enumerate((2, 4, 4, 4)):
        name = f'vq.learned_up_l1.{index}'
        reference_input = ref['embedding'] if index == 0 else ref[f'cascade.{index - 1}']
        same_input = model.codec.conv(name, mx.array(reference_input), stride=stride, transpose=True)
        x = model.codec.conv(name, x, stride=stride, transpose=True)
        expected = ref[f'cascade.{index}']
        results[f'cascade.{index}'] = {'same_input': compare(same_input, expected),
                                      'propagated': compare(x, expected)}
    reference_input = mx.array(ref['cascade.3'])
    conv = model.codec.conv('vq.quant_resi.0', reference_input, padding=1)
    results['residual.conv'] = {'same_input': compare(conv, ref['residual.conv'])}
    results['refined'] = {'same_input': compare(model.codec.phi(reference_input, 0), ref['refined']),
                          'propagated': compare(model.codec.phi(x, 0), ref['refined'])}
    after = {str(p): digest(p) for p in watched}
    if before != after:
        raise ValueError('source or input changed during diagnostic')
    with args.output.open('x') as out:
        json.dump({'input_sha256': before, 'atol': 5e-4, 'rtol': 2e-4, 'results': results,
                   'scope': 'same-input CUDA versus independent MLX cascade; diagnostic only'},
                  out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
