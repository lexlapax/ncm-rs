"""Screen simple FP32 FMA reduction orders against retained CUDA residuals."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


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
    parser.add_argument('--mlx-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    manifest = args.artifact / 'manifest.json'
    codec = args.artifact / 'codec.safetensors'
    if digest(manifest) != '0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1':
        raise ValueError('unexpected artifact manifest')
    metadata = json.loads(manifest.read_bytes())
    if digest(codec) != metadata['files']['codec.safetensors']['sha256']:
        raise ValueError('unexpected codec payload')
    if digest(args.reference) != 'b320d70ba0f2393cd1276200db064d477d000c87241c6541c680699391f8b085':
        raise ValueError('unexpected CUDA reference')
    watched = [Path(__file__).resolve(), manifest.resolve(), codec.resolve(),
               args.reference.resolve(), args.mlx_capture.resolve()]
    before = {str(p): digest(p) for p in watched}
    ref, metal, weights = load_file(args.reference), load_file(args.mlx_capture), load_file(codec)
    x, conv, refined = ref['cascade.3'], ref['residual.conv'], ref['refined']
    w, bias = weights['vq.quant_resi.0.weight'], weights['vq.quant_resi.0.bias']
    if (x.shape != (128, 128, 384) or conv.shape != x.shape or refined.shape != x.shape
            or w.shape != (384, 384, 3) or bias.shape != (384,)
            or set(metal) != {'cascade.3', 'refined'} or not np.array_equal(x, metal['cascade.3'])
            or metal['refined'].shape != x.shape):
        raise ValueError('unexpected residual tensor shapes or inputs')
    count = x.size
    rng = np.random.default_rng(4217)
    selected = np.concatenate((np.argpartition(np.abs(refined).reshape(-1), 128)[:128],
                               rng.choice(count, size=128, replace=False)))
    batch, pos, channel = np.unravel_index(selected, x.shape)
    patches = np.zeros((len(selected), 3, 384), dtype=np.float32)
    for k in range(3):
        t = pos + k - 1
        valid = (t >= 0) & (t < 128)
        patches[valid, k] = x[batch[valid], t[valid], :]
    kernels = w[channel].transpose(0, 2, 1)
    target_conv = conv.reshape(-1)[selected]
    target_blend = refined.reshape(-1)[selected]
    target_metal = metal['refined'].reshape(-1)[selected]
    input_value = x.reshape(-1)[selected]
    bias_value = bias[channel]
    libm = ctypes.CDLL('libm.so.6')
    fmaf = libm.fmaf
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fmaf.restype = ctypes.c_float
    results = {}
    fma_checks = 0
    for order in ('kernel-major', 'channel-major'):
        a = patches if order == 'kernel-major' else patches.transpose(0, 2, 1)
        b = kernels if order == 'kernel-major' else kernels.transpose(0, 2, 1)
        a = a.reshape(len(selected), -1).astype(np.float64)
        b = b.reshape(len(selected), -1).astype(np.float64)
        for lanes in (1, 2, 4, 8, 16, 32):
            for bias_first in (False, True):
                accum = np.zeros((lanes, len(selected)), dtype=np.float32)
                if bias_first:
                    accum[0] = bias_value
                for j in range(a.shape[1]):
                    lane = j % lanes
                    if j in (0, 1, 31, 255, 1151):
                        for i in range(0, len(selected), 16):
                            actual = np.float32(float(a[i, j]) * float(b[i, j]) + float(accum[lane, i]))
                            if actual.view(np.uint32) != np.float32(fmaf(a[i, j], b[i, j], float(accum[lane, i]))).view(np.uint32):
                                raise ValueError('FP64 FMA emulation differs from libm fmaf')
                            fma_checks += 1
                    accum[lane] = (a[:, j] * b[:, j] + accum[lane].astype(np.float64)).astype(np.float32)
                for reduction in ('sequential', 'tree'):
                    if reduction == 'sequential':
                        total = accum[0].copy()
                        for lane in range(1, lanes):
                            total = (total + accum[lane]).astype(np.float32)
                    else:
                        partial = accum.copy()
                        step = 1
                        while step < lanes:
                            for lane in range(0, lanes, 2 * step):
                                partial[lane] = (partial[lane] + partial[lane + step]).astype(np.float32)
                            step *= 2
                        total = partial[0]
                    if not bias_first:
                        total = (total + bias_value).astype(np.float32)
                    blend = (input_value * np.float32(0.5) + total * np.float32(0.5)).astype(np.float32)
                    difference = np.abs(blend.astype(np.float64) - target_blend.astype(np.float64))
                    band = 5e-4 + 2e-4 * np.abs(target_blend.astype(np.float64))
                    key = f'{order}/lanes-{lanes}/bias-{int(bias_first)}/{reduction}'
                    results[key] = {'conv_bit_equal': int(np.count_nonzero(total.view(np.uint32) == target_conv.view(np.uint32))),
                                    'blend_bit_equal': int(np.count_nonzero(blend.view(np.uint32) == target_blend.view(np.uint32))),
                                    'blend_outside_band': int(np.count_nonzero(difference > band)),
                                    'blend_max_abs': float(difference.max())}
    difference = np.abs(target_metal.astype(np.float64) - target_blend.astype(np.float64))
    after = {str(p): digest(p) for p in watched}
    if before != after:
        raise ValueError('source or inputs changed')
    payload = {'sample_count': len(selected), 'selection': '128 smallest absolute CUDA blended outputs plus 128 deterministic random positions',
               'selection_seed': 4217, 'fma_checked_operations': fma_checks, 'input_sha256': before,
               'mlx_blend_bit_equal': int(np.count_nonzero(target_metal.view(np.uint32) == target_blend.view(np.uint32))),
               'mlx_blend_outside_band': int(np.count_nonzero(difference > 5e-4 + 2e-4 * np.abs(target_blend.astype(np.float64)))),
               'results': results, 'scope': 'small sampled diagnostic; emulated FP32 FMA was spot-checked against libm fmaf'}
    with args.output.open('x') as out:
        json.dump(payload, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
