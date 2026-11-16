"""Screen equivalent MLX residual convolution formulations on captured inputs."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_codec import prepare_device
from mlx_codec_assets import save_tensors, tensors
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
        raise ValueError('invalid output shape or values')
    ref = expected.astype(np.float64)
    delta = np.abs(value.astype(np.float64) - ref)
    norm = delta / (5e-4 + 2e-4 * np.abs(ref))
    return {'elements': int(value.size), 'different_bits': int(np.count_nonzero(value.view(np.uint32) != expected.view(np.uint32))),
            'outside_band': int(np.count_nonzero(norm > 1)), 'max_abs': float(delta.max()),
            'max_normalized': float(norm.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(16, 128), default=128)
    args = parser.parse_args()
    if args.output_dir.exists() or digest(args.reference) != args.reference_sha256:
        raise ValueError('output directory must be new and reference hash must match')
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
    source = ref['cascade.3']
    if source.shape != (128, 128, 384) or ref['residual.conv'].shape != source.shape or ref['refined'].shape != source.shape:
        raise ValueError('unexpected CUDA reference shapes')
    raw = source[:args.batch]
    x = mx.array(raw)
    w = model.codec.w['vq.quant_resi.0.weight']
    bias = model.codec.w['vq.quant_resi.0.bias']
    if tuple(w.shape) != (384, 384, 3) or tuple(bias.shape) != (384,):
        raise ValueError('unexpected convolution weight shapes')
    weights_2d = w.transpose(0, 2, 1).reshape(384, 1, 3, 384)
    padded = mx.pad(x, ((0, 0), (1, 1), (0, 0)))

    def baseline():
        return model.codec.conv('vq.quant_resi.0', x, padding=1)

    def conv2d():
        return mx.conv2d(x[:, None, :, :], weights_2d, padding=(0, 1))[:, 0, :, :] + bias

    def three_gemm():
        parts = [padded[:, k:k + 128, :] @ w[:, :, k].T for k in range(3)]
        return parts[0] + parts[1] + parts[2] + bias

    args.output_dir.mkdir(parents=True)
    results = {}
    for name, operation in (('conv1d', baseline), ('conv2d', conv2d), ('three_gemm', three_gemm)):
        times = []
        for iteration in range(8):
            mx.synchronize()
            start = time.perf_counter()
            value = operation()
            mx.eval(value)
            elapsed = time.perf_counter() - start
            if iteration >= 3:
                times.append(elapsed)
        blend = np.asarray(0.5 * x + 0.5 * value)
        capture = args.output_dir / f'{name}.safetensors'
        save_tensors(capture, {'cascade.3': raw, 'refined': blend})
        results[name] = {'conv': compare(value, ref['residual.conv'][:args.batch]),
                         'blend': compare(blend, ref['refined'][:args.batch]),
                         'median_ms': statistics.median(times) * 1000, 'trials_seconds': times,
                         'capture_sha256': digest(capture)}
    after = {str(p): digest(p) for p in watched}
    if before != after:
        raise ValueError('source or inputs changed')
    with (args.output_dir / 'summary.json').open('x') as out:
        json.dump({'input_sha256': before, 'batch': args.batch, 'atol': 5e-4, 'rtol': 2e-4, 'results': results,
                   'scope': 'same-input operator screen, 3 warmups and 5 synchronized calls; not a model gate'},
                  out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
