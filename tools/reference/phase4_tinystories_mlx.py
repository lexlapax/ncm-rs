"""Experimental TinyStories Python MLX adapter; pinned CUDA Python remains the oracle."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_codec import Codec, prepare_device
from mlx_codec_assets import digest, save_tensors, tensors
from mlx_operators import execute
from phase3_mlx_generator import Generator

MANIFEST_SHA256 = '0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1'


class TinyCodec(Codec):
    def up(self, x, level):
        if level == 0:
            for i, stride in enumerate((2, 4, 4, 4)):
                x = self.conv(f'vq.learned_up_l1.{i}', x, stride=stride, transpose=True)
            return x
        if level == len(self.levels) - 1:
            return x
        return execute({'kind': 'linear1d', 'length': self.length}, {'x': x})


class TinyGenerator(Generator):
    def __init__(self, root):
        manifest_raw = (root / 'manifest.json').read_bytes()
        if digest(manifest_raw) != MANIFEST_SHA256:
            raise ValueError('unexpected TinyStories manifest')
        manifest = json.loads(manifest_raw)
        def payload(name):
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('invalid artifact file')
            value = path.read_bytes()
            item = manifest['files'][name]
            if len(value) != item['bytes'] or digest(value) != item['sha256']:
                raise ValueError(f'artifact mismatch: {name}')
            return value
        config = json.loads(payload('config.json'))
        self.cfg = config['model']
        codec_config = config['tokenizer']['vqvae']
        self.levels = codec_config['vector_quantizer_config']['levels']['manual_levels']
        codec = tensors(payload('codec.safetensors'))
        generator = tensors(payload('generator.safetensors'))
        self.codec = TinyCodec(codec_config, {k: mx.array(v) for k, v in codec.items()})
        self.w = {k: mx.array(v) for k, v in generator.items() if v.dtype == np.float32}
        self.unigram = mx.array(tensors(payload('inference.safetensors'))['level0_unigram'])
        self.geometry = []
        p = self.cfg['prefix_len']
        half = self.cfg['n_embd'] // self.cfg['n_head'] // 2
        for length in self.levels[1:]:
            mask = np.zeros((1, 1, p + length, p + length), dtype=np.float32)
            mask[:, :, :p, p:] = -np.inf
            positions = np.concatenate((np.arange(p), np.arange(length))).astype(np.float32)
            frequency = np.float32(self.cfg['rope_base']) ** (np.arange(half, dtype=np.float32) / np.float32(half))
            angles = positions[:, None] / frequency[None, :]
            cosine, sine = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
            self.geometry.append((mx.array(mask), mx.array(np.concatenate((cosine, cosine), axis=-1)),
                                  mx.array(np.concatenate((sine, sine), axis=-1))))
        mx.eval(*self.codec.w.values(), *self.w.values(), self.unigram,
                *(value for group in self.geometry for value in group))

    def run(self, batch, seed, forced=None, observe=None):
        mx.random.seed(seed)
        prefix_ids = mx.zeros((batch, 1), dtype=mx.uint32)
        prefix = self.linear('transformer.wte', self.codec.w['encoder.wte.weight'][prefix_ids])
        if forced is None:
            first = mx.random.categorical(mx.log(self.unigram), shape=(batch,)).reshape(batch, 1)
            selected = [first]
        else:
            if forced.shape != (batch, sum(self.levels)):
                raise ValueError('invalid teacher-forced codes')
            selected = [mx.array(forced[:, :1].astype(np.uint32))]
        cumulative = None
        offset = 1
        for level, length in enumerate(self.levels[1:]):
            book = self.codec.w[f'vq.embeddings.{level}.weight']
            up = self.codec.up(book[selected[-1]], level)
            contribution = self.codec.phi(up, level)
            cumulative = contribution if cumulative is None else cumulative + contribution
            latent = execute({'kind': 'area1d', 'length': length}, {'x': cumulative})
            target = self.linear('transformer.wte', latent) + self.w['transformer.wle.weight'][level]
            hidden = self.level(prefix, target, level)
            head = f'lm_heads.{level}'
            projected = hidden @ self.w[head + '.proj.weight'].T
            next_book = self.codec.w[f'vq.embeddings.{level + 1}.weight']
            norms = mx.sum(mx.square(next_book), axis=-1)
            logits = (2 * (projected @ next_book.T) - norms) / mx.mean(norms) * mx.exp(self.w[head + '.log_inv_T'])
            if observe is not None:
                observe(f'generator.level_{level + 1:02d}.logits', logits)
            if forced is None:
                chosen = mx.random.categorical(logits, axis=-1)
            else:
                chosen = mx.array(forced[:, offset:offset + length].astype(np.uint32))
            selected.append(chosen)
            offset += length
        codes = mx.concatenate(selected, axis=1)
        final_book = self.codec.w[f'vq.embeddings.{len(self.levels)-1}.weight']
        latent = cumulative + self.codec.phi(self.codec.up(final_book[selected[-1]], len(self.levels)-1), len(self.levels)-1)
        hidden = self.codec.block(latent)
        hidden = self.codec.norm('decoder.ln_f', hidden)
        logits = self.codec.linear('decoder.output_layer', hidden)
        ids = mx.argmax(logits, axis=-1)
        if observe is not None:
            observe('generator.final_logits', logits)
            observe('generator.output_ids', ids)
        mx.eval(codes, ids)
        return codes, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--adapted-trace', type=Path, help='write a merged trace for native MLX comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--wall-markers', action='store_true', help='diagnostic request timestamps')
    parser.add_argument('--diagnostic-cache-mib', type=int, help='diagnostic process-local free-cache limit')
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.batch <= 128
            or (args.diagnostic_cache_mib is not None and not 1 <= args.diagnostic_cache_mib <= 16384)
            or (args.adapted_trace is not None and (args.reference is None or args.adapted_trace.exists()))):
        raise ValueError('output must be new and batch in 1..128')
    prepare_device()
    previous_cache_limit = (mx.set_cache_limit(args.diagnostic_cache_mib * 1024 * 1024)
                            if args.diagnostic_cache_mib is not None else None)
    model = TinyGenerator(args.artifact)
    reference = tensors(args.reference.read_bytes()) if args.reference else None
    stats = {}
    adapted = {}
    def observe(name, value):
        actual = np.asarray(value)
        expected = reference[name]
        if args.adapted_trace is not None:
            adapted[name] = actual.copy()
        if actual.shape != expected.shape:
            raise ValueError(f'{name} shape mismatch')
        if np.issubdtype(expected.dtype, np.integer):
            stats[name] = {'elements': expected.size, 'differences': int(np.count_nonzero(actual != expected))}
        else:
            diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            stats[name] = {'elements': expected.size, 'max_abs': float(diff.max()),
                           'accumulated_outside': int(np.count_nonzero(diff > 5e-4 + 2e-4 * np.abs(expected)))}
            if name.startswith('generator.level_'):
                stats[name]['argmax_differences'] = int(np.count_nonzero(actual.argmax(-1) != expected.argmax(-1)))
    forced = reference['generator.forced_codes'] if reference else None
    if forced is not None and len(forced) != args.batch:
        raise ValueError('reference batch mismatch')
    trials = []
    request_wall_ns = []
    for trial in range(35 if args.bench else 1):
        mx.synchronize()
        begin_wall = time.time_ns() if args.wall_markers else None
        start = time.perf_counter()
        codes, ids = model.run(args.batch, args.seed, forced, observe if reference else None)
        codes_list, ids_list = codes.tolist(), ids.tolist()
        elapsed = time.perf_counter() - start
        if args.wall_markers:
            end_wall = time.time_ns()
            if end_wall < begin_wall:
                raise ValueError('wall clock moved backward during request')
            request_wall_ns.append([begin_wall, end_wall, end_wall])
        if trial >= 5:
            trials.append(elapsed)
        del codes, ids
    report = {'batch': args.batch, 'seed': args.seed, 'runtime': 'experimental Python MLX adaptation',
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
              'stats': stats, 'codes': codes_list, 'ids': ids_list, 'warmups': 5 if args.bench else 0,
              'trials_seconds': trials, 'mlx_memory': {'active': mx.get_active_memory(),
              'peak': mx.get_peak_memory(), 'cache': mx.get_cache_memory()}}
    if args.wall_markers:
        report['request_wall_ns'] = request_wall_ns
    if args.diagnostic_cache_mib is not None:
        report['diagnostic_cache_limit_bytes'] = args.diagnostic_cache_mib * 1024 * 1024
        report['previous_cache_limit_bytes'] = previous_cache_limit
    if args.adapted_trace is not None:
        reference.update(adapted)
        save_tensors(args.adapted_trace, reference)
    args.output.write_text(json.dumps(report, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
