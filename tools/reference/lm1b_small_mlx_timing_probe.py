"""Time the Python MLX LM1B-small adaptation for one fixed batch.

The forward pass and FP64 decoding penalties are those of the qualified development reference
`lm1b_small_mlx_reference.py`. Its exact controlled sampler is a pure-Python loop over every
8,192-way distribution, which is not a defensible speed baseline. This probe samples on-device
with `mx.random.categorical`, as the TinyStories Python MLX baseline does. The random streams
differ from Rust's, so outputs are not compared; only timing and memory are.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from lm1b_small_mlx_reference import Lm1bGenerator
from mlx_operators import execute


class TimingGenerator(Lm1bGenerator):
    def sample_native(self, batch, seed):
        mx.random.seed(seed)
        prefix_ids = mx.full((batch, 1), 101, dtype=mx.uint32)
        prefix = self.linear('transformer.wte', self.codec.w['encoder.wte.weight'][prefix_ids])
        log_initial = mx.log(mx.array(self.initial_weights, dtype=mx.float32))
        initial = mx.random.categorical(mx.broadcast_to(log_initial, (batch, log_initial.size)), axis=-1)
        codes = []
        cumulative = None
        for level, length in enumerate(self.levels):
            if level == 0:
                chosen = initial.reshape(batch, 1).astype(mx.uint32)
            else:
                latent = execute({'kind': 'area1d', 'length': length}, {'x': cumulative})
                conv = f'post_upsample_convs.{level}'
                delta = mx.conv1d(latent, self.w[conv + '.weight'].transpose(0, 2, 1), padding=1)
                target = self.linear('transformer.wte', latent + (delta + self.w[conv + '.bias']))
                hidden = self.level(prefix, target + self.w['transformer.wle.weight'][level], level)
                logits = hidden @ self.w['lm_heads.head.weight'].T + self.w[f'lm_heads.biases.{level}']
                chosen = mx.random.categorical(logits, axis=-1).astype(mx.uint32)
            codes.append(chosen)
            upsampled = self.codec.up(self.codec.w[f'vq.embeddings.{level}.weight'][chosen], level)
            contribution = self.codec.phi(upsampled, level)
            cumulative = contribution if cumulative is None else cumulative + contribution
        logits = np.asarray(self.codec.logits(cumulative))
        if logits.shape != (batch, 128, 30528) or not np.isfinite(logits).all():
            raise ValueError('invalid decoder logits')
        ids = np.array([self.penalized_ids(row) for row in logits], dtype=np.uint32)
        return np.asarray(mx.concatenate(codes, axis=1)).astype(np.uint32), ids


def memory_counters():
    get = lambda name: int(getattr(mx, name, None)() if hasattr(mx, name) else getattr(mx.metal, name)())
    return {'active': get('get_active_memory'), 'peak': get('get_peak_memory'), 'cache': get('get_cache_memory')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('rotary', type=Path)
    parser.add_argument('batch', type=int)
    parser.add_argument('warmups', type=int)
    parser.add_argument('trials', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32 or args.trials < 1 or args.warmups < 0:
        parser.error('batch must be 1..32 and trials positive')
    if args.output.exists() or args.output.is_symlink():
        parser.error('output must be a new file')
    start = time.perf_counter()
    model = TimingGenerator(args.artifact, args.rotary)
    load_seconds = time.perf_counter() - start
    seconds, signatures = [], []
    for trial in range(args.warmups + args.trials):
        start = time.perf_counter()
        codes, ids = model.sample_native(args.batch, 11_000 + trial)
        elapsed = time.perf_counter() - start
        if codes.shape != (args.batch, 641) or ids.shape != (args.batch, 128):
            raise ValueError('unexpected output shape')
        if trial >= args.warmups:
            seconds.append(elapsed)
            signatures.append(hashlib.sha256(codes.astype('<u4').tobytes() + ids.astype('<u4').tobytes()).hexdigest())
    result = {'runtime': 'python', 'batch': args.batch, 'warmups': args.warmups, 'trials': args.trials,
              'scope': 'generation, decoding penalties and code/ID readback; model load excluded',
              'sampler': 'mx.random.categorical on device; reference forward pass and FP64 penalties',
              'load_seconds': load_seconds, 'seconds': seconds, 'signatures': signatures,
              'mlx_memory': memory_counters(), 'mlx_version': mx.__version__}
    with args.output.open('x') as file:
        json.dump(result, file, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
