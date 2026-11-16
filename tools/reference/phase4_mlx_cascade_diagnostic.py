"""Compare independently recomputed MLX reconstruction levels to CUDA traces."""
import argparse
import hashlib
import importlib.metadata
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    root = Path(__file__).resolve().parents[2]
    watched = [Path(__file__).resolve(), args.reference.resolve(),
               *(root / 'tools/reference' / name for name in (
                   'phase4_tinystories_mlx.py', 'phase3_mlx_generator.py',
                   'mlx_codec.py', 'mlx_codec_assets.py', 'mlx_operators.py'))]
    watched += sorted(p.resolve() for p in args.artifact.rglob('*') if p.is_file())
    before = {str(p): digest(p) for p in watched}
    prepare_device()
    model = TinyGenerator(args.artifact)
    reference = tensors(args.reference.read_bytes())
    forced = reference['generator.forced_codes']
    levels = model.levels
    if (forced.dtype.kind not in 'iu' or forced.ndim != 2 or not 1 <= forced.shape[0] <= 16
            or forced.shape[1] != sum(levels) or np.any(forced < 0) or np.any(forced >= 4096)):
        raise ValueError('invalid forced code indices')
    offset = 0
    cumulative = None
    results = {}
    for level, length in enumerate(levels):
        indices = mx.array(forced[:, offset:offset + length].astype(np.uint32))
        book = model.codec.w[f'vq.embeddings.{level}.weight']
        contribution = model.codec.phi(model.codec.up(book[indices], level), level)
        cumulative = contribution if cumulative is None else cumulative + contribution
        actual = np.asarray(cumulative)
        expected = reference[f'reconstruction.level_{level:02d}']
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('invalid reconstruction values or shape')
        ref = expected.astype(np.float64)
        diff = np.abs(actual.astype(np.float64) - ref)
        normalized = diff / (5e-4 + 2e-4 * np.abs(ref))
        results[f'reconstruction.level_{level:02d}'] = {
            'elements': int(actual.size), 'outside_band': int(np.count_nonzero(normalized > 1)),
            'max_abs': float(diff.max()), 'max_normalized': float(normalized.max()),
        }
        offset += length
    after = {str(p): digest(p) for p in watched}
    if before != after:
        raise ValueError('source or input changed during diagnostic')
    report = {'batch': int(forced.shape[0]), 'atol': 5e-4, 'rtol': 2e-4,
              'reference': str(args.reference), 'input_sha256': before,
              'mlx': importlib.metadata.version('mlx'), 'scope': 'independent MLX reconstruction; fixed CUDA codes',
              'results': results}
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
