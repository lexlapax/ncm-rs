"""Offline MLX adaptation of pinned Sudoku-easy generation for same-device comparison.

The original PyTorch CUDA reference remains the quality oracle. This adapter is
not qualified until its codes and IDs have been compared with that oracle.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_codec import Codec, prepare_device
from mlx_codec_assets import ROOT, digest, load_bundle, tensors
from mlx_operators import execute


class Generator:
    def __init__(self, root):
        # load_bundle authenticates the fixed easy manifest and all payloads,
        # including every generator tensor, against the Phase 1 anchor.
        codec_config, codec_host = load_bundle(root)
        anchor = json.loads((ROOT / 'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json').read_bytes())
        def verified_bytes(name):
            path = root / name
            record = anchor['files'][name]
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes']:
                raise ValueError(f'invalid {name} file')
            data = path.read_bytes()
            if len(data) != record['bytes'] or digest(data) != record['sha256']:
                raise ValueError(f'{name} differs from the pinned artifact')
            return data
        config = json.loads(verified_bytes('config.json'))
        self.cfg = config['model']
        self.levels = codec_config['vector_quantizer_config']['levels']['manual_levels']
        generator_host = tensors(verified_bytes('generator.safetensors'))
        self.codec = Codec(codec_config, {n: mx.array(v) for n, v in codec_host.items()})
        self.w = {n: mx.array(v) for n, v in generator_host.items() if v.dtype == np.float32}
        mx.eval(*self.codec.w.values(), *self.w.values())
        self.geometry = []
        p = self.cfg['prefix_len']
        half = self.cfg['n_embd'] // self.cfg['n_head'] // 2
        for length in self.levels:
            mask = np.zeros((1, 1, p + length, p + length), dtype=np.float32)
            mask[:, :, :p, p:] = -np.inf
            positions = np.concatenate((np.arange(p), np.arange(length))).astype(np.float32)
            frequency = np.float32(self.cfg['rope_base']) ** (np.arange(half, dtype=np.float32) / np.float32(half))
            angles = positions[:, None] / frequency[None, :]
            cosine = np.cos(angles).astype(np.float32)
            sine = np.sin(angles).astype(np.float32)
            self.geometry.append((mx.array(mask), mx.array(np.concatenate((cosine, cosine), axis=-1)),
                                  mx.array(np.concatenate((sine, sine), axis=-1))))
        mx.eval(*(v for group in self.geometry for v in group))

    def linear(self, name, x):
        return x @ self.w[name + '.weight'].T + self.w[name + '.bias']

    def norm(self, name, x, rms=False):
        kind = 'rms_norm' if rms else 'layer_norm'
        y = execute({'kind': kind, 'epsilon': 1e-6}, {'x': x}) * self.w[name + '.weight']
        return y if rms else y + self.w[name + '.bias']

    def level(self, prefix, target, level):
        mask, cosine, sine = self.geometry[level]
        x = mx.concatenate([prefix, target], axis=1)
        batch, seq, width = x.shape
        heads = self.cfg['n_head']
        for i in range(self.cfg['n_layer']):
            name = f'transformer.blocks.{i}'
            norm = self.norm(name + '.ln_1', x)
            qkv = self.linear(name + '.attn.c_attn', norm)
            q, k, v = [part.reshape(batch, seq, heads, -1).transpose(0, 2, 1, 3)
                       for part in mx.split(qkv, 3, axis=2)]
            q = self.norm(name + '.attn.q_norm', q, rms=True)
            k = self.norm(name + '.attn.k_norm', k, rms=True)
            q = execute({'kind': 'rope'}, {'x': q, 'cos': cosine, 'sin': sine})
            k = execute({'kind': 'rope'}, {'x': k, 'cos': cosine, 'sin': sine})
            attn = execute({'kind': 'attention'}, {'q': q, 'k': k, 'v': v, 'mask': mask})
            attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq, width)
            x = x + self.linear(name + '.attn.c_proj', attn)
            norm = self.norm(name + '.ln_2', x)
            hidden = self.linear(name + '.mlp.c_fc', norm)
            hidden = 0.5 * hidden * (1 + mx.erf(hidden / math.sqrt(2)))
            x = x + self.linear(name + '.mlp.c_proj', hidden)
        return self.norm('transformer.ln_f', x)[:, self.cfg['prefix_len']:, :]

    def generate(self, rows):
        batch = len(rows)
        ids = mx.array(rows, dtype=mx.uint32)
        prefix = self.linear('transformer.wte', self.codec.w['encoder.wte.weight'][ids])
        cumulative = mx.zeros((batch, self.codec.length, self.codec.width), dtype=mx.float32)
        selected = []
        for level, length in enumerate(self.levels):
            if level == 0:
                latent = mx.broadcast_to(self.w['l0_queries'], (batch, length, self.codec.width))
            else:
                book = self.codec.w[f'vq.embeddings.{level-1}.weight']
                up = self.codec.up(book[selected[-1]], level - 1)
                cumulative = cumulative + self.codec.phi(up, level - 1)
                latent = execute({'kind': 'area1d', 'length': length}, {'x': cumulative})
            target = self.linear('transformer.wte', latent) + self.w['transformer.wle.weight'][level]
            logits = self.linear(f'lm_heads.{level}', self.level(prefix, target, level))
            selected.append(mx.argmax(logits, axis=-1))
        codes = mx.concatenate(selected, axis=1)
        _, output, _ = self.codec.decode(codes)
        mx.eval(codes, output)
        return codes, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('output must be new')
    raw = args.inputs.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128 or any(not isinstance(r, list) or len(r) != 91 or any(type(v) is not int or not 0 <= v < 12 for v in r) for r in rows):
        raise ValueError('invalid input rows')
    # Match the native harness: parse/flatten host inputs before timed calls;
    # each timed call still uploads them to the explicitly selected GPU.
    rows = np.asarray(rows, dtype=np.uint32)
    prepare_device()
    generator = Generator(args.artifact)
    expected = None
    trials = []
    for trial in range(35 if args.bench else 1):
        mx.synchronize()
        start = time.perf_counter()
        codes, ids = generator.generate(rows)
        codes_list, ids_list = codes.tolist(), ids.tolist()
        elapsed = time.perf_counter() - start
        payload = np.asarray(codes_list, dtype='<i8').tobytes() + np.asarray(ids_list, dtype='<i8').tobytes()
        fingerprint = hashlib.sha256(payload).hexdigest()
        if expected is not None and fingerprint != expected: raise ValueError('outputs changed across trials')
        expected = fingerprint
        if trial >= 5: trials.append(elapsed)
        del codes, ids
    report = {'batch': len(rows), 'inputs_sha256': digest(raw), 'outputs_sha256_i64le': expected,
              'codes': codes_list, 'ids': ids_list, 'runtime': 'python-mlx-adapted',
              'scope': 'generation, codec decode and synchronized code/ID readback; excludes model load',
              'warmups': 5 if args.bench else 0, 'trials': trials,
              'mlx_peak_bytes': mx.get_peak_memory(), 'mlx_active_bytes': mx.get_active_memory(),
              'mlx_cache_bytes': mx.get_cache_memory()}
    with args.output.open('x') as file: json.dump(report, file, separators=(',', ':')); file.write('\n')


if __name__ == '__main__':
    main()
