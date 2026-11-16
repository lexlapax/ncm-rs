"""Independent Python MLX XSum top-1 adapter; pinned Python CUDA remains the oracle."""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_codec import Codec, prepare_device
from mlx_codec_assets import digest, tensors
from mlx_operators import execute
from phase3_mlx_generator import Generator
from xsum_numpy_selection import select

MANIFEST_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
INPUT_SHA = 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf'


class XsumCodec(Codec):
    def up(self, x, level):
        if level == 0:
            x = self.conv('vq.learned_up_l1.0', x, stride=3, transpose=True)
            x = execute({'kind': 'group_norm', 'groups': 8, 'epsilon': 1e-5}, {'x': x})
            return self.conv('vq.learned_up_l1.2', x, stride=4, transpose=True)
        if level == len(self.levels) - 1:
            return x
        return execute({'kind': 'linear1d', 'length': self.length}, {'x': x})


class XsumGenerator(Generator):
    def __init__(self, root):
        manifest_raw = (root / 'manifest.json').read_bytes()
        if digest(manifest_raw) != MANIFEST_SHA:
            raise ValueError('unexpected XSum manifest')
        manifest = json.loads(manifest_raw)
        def payload(name):
            path = root / name
            item = manifest['files'][name]
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes']:
                raise ValueError(f'invalid artifact {name}')
            data = path.read_bytes()
            if digest(data) != item['sha256']:
                raise ValueError(f'artifact mismatch: {name}')
            return data
        config = json.loads(payload('config.json'))
        self.cfg = config['model']
        codec_config = config['tokenizer']['vqvae']
        self.levels = codec_config['vector_quantizer_config']['levels']['manual_levels']
        codec = tensors(payload('codec.safetensors'))
        generator = tensors(payload('generator.safetensors'))
        inference = tensors(payload('inference.safetensors'))
        self.codec = XsumCodec(codec_config, {k: mx.array(v) for k, v in codec.items()})
        self.w = {k: mx.array(v) for k, v in generator.items() if v.dtype == np.float32}
        self.unigram = np.asarray(inference['token_unigram'], dtype=np.float32)
        self.head_norms = []
        for level in range(len(self.levels)):
            book = self.codec.w[f'vq.embeddings.{level}.weight']
            norms = mx.sum(mx.square(book), axis=-1)
            self.head_norms.append((norms, mx.mean(norms), mx.exp(self.w[f'lm_heads.{level}.log_inv_T'])))
        self.geometry = []
        for length in self.levels:
            p = self.cfg['prefix_len']
            mask = np.zeros((1, 1, p + length, p + length), dtype=np.float32)
            mask[:, :, :p, p:] = -np.inf
            cosine = np.concatenate((inference['rope_cos'][:p], inference['rope_cos'][:length]), axis=0)
            sine = np.concatenate((inference['rope_sin'][:p], inference['rope_sin'][:length]), axis=0)
            self.geometry.append((mx.array(mask), mx.array(np.concatenate((cosine, cosine), axis=-1)),
                                  mx.array(np.concatenate((sine, sine), axis=-1))))
        mx.eval(*self.codec.w.values(), *self.w.values(), *(v for g in self.geometry for v in g),
                *(v for g in self.head_norms for v in g))

    def level(self, prefix, target, level, padding):
        mask, cosine, sine = self.geometry[level]
        mask = mask + padding
        x = mx.concatenate([prefix, target], axis=1)
        batch, seq, width = x.shape
        heads = self.cfg['n_head']
        for i in range(self.cfg['n_layer']):
            name = f'transformer.blocks.{i}'
            norm = self.norm(name + '.ln_1', x)
            qkv = self.linear(name + '.attn.c_attn', norm)
            q, k, v = [part.reshape(batch, seq, heads, -1).transpose(0, 2, 1, 3) for part in mx.split(qkv, 3, axis=2)]
            q = self.norm(name + '.attn.q_norm', q, rms=True)
            k = self.norm(name + '.attn.k_norm', k, rms=True)
            q = execute({'kind': 'rope'}, {'x': q, 'cos': cosine, 'sin': sine})
            k = execute({'kind': 'rope'}, {'x': k, 'cos': cosine, 'sin': sine})
            attention = execute({'kind': 'attention'}, {'q': q, 'k': k, 'v': v, 'mask': mask})
            attention = attention.transpose(0, 2, 1, 3).reshape(batch, seq, width)
            x = x + self.linear(name + '.attn.c_proj', attention)
            hidden = self.linear(name + '.mlp.c_fc', self.norm(name + '.ln_2', x))
            hidden = 0.5 * hidden * (1 + mx.erf(hidden / math.sqrt(2)))
            x = x + self.linear(name + '.mlp.c_proj', hidden)
        return self.norm('transformer.ln_f', x)[:, self.cfg['prefix_len']:, :]

    def generate(self, rows):
        batch = len(rows)
        ids = mx.array(rows, dtype=mx.uint32)
        prefix = self.linear('transformer.wte', self.codec.w['encoder.wte.weight'][ids])
        selected = []
        cumulative = None
        for level, length in enumerate(self.levels):
            if level == 0:
                latent = mx.broadcast_to(self.w['l0_queries'], (batch, length, self.codec.width))
            else:
                latent = execute({'kind': 'area1d', 'length': length}, {'x': cumulative})
            target = self.linear('transformer.wte', latent) + self.w['transformer.wle.weight'][level]
            pad = np.zeros((batch, 1, 1, 512 + length), dtype=np.float32)
            pad[:, 0, 0, :512] = np.where(rows == 50257, -np.inf, 0.0)
            hidden = self.level(prefix, target, level, mx.array(pad))
            book = self.codec.w[f'vq.embeddings.{level}.weight']
            norms, mean_norm, scale = self.head_norms[level]
            projected = hidden @ self.w[f'lm_heads.{level}.proj.weight'].T
            logits = (2 * (projected @ book.T) - norms) / mean_norm * scale
            code = mx.argmax(logits, axis=-1)
            selected.append(code)
            contribution = self.codec.phi(self.codec.up(book[code], level), level)
            cumulative = contribution if cumulative is None else cumulative + contribution
        codes = mx.concatenate(selected, axis=1)
        hidden = self.codec.norm('decoder.ln_f', self.codec.block(cumulative))
        logits = self.codec.linear('decoder.output_layer', hidden)
        mx.eval(codes, logits)
        host = np.asarray(logits)
        output = select(host, self.unigram)
        return codes.tolist(), output.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.batch <= 32:
        raise ValueError('new output and batch 1..32 required')
    raw = args.inputs.read_bytes()
    if digest(raw) != INPUT_SHA:
        raise ValueError('unexpected input')
    rows = np.asarray(json.loads(raw)[:args.batch], dtype=np.uint32)
    if rows.shape != (args.batch, 512):
        raise ValueError('unexpected input shape')
    prepare_device()
    model = XsumGenerator(args.artifact)
    timings = []
    signature = None
    for trial in range(7 if args.bench else 1):
        mx.synchronize()
        start = time.perf_counter()
        codes, ids = model.generate(rows)
        elapsed = time.perf_counter() - start
        current = hashlib.sha256(np.asarray(codes, dtype='<u4').tobytes() + np.asarray(ids, dtype='<u4').tobytes()).hexdigest()
        if signature is not None and signature != current:
            raise ValueError('output changed between trials')
        signature = current
        if args.bench and trial >= 2:
            timings.append(elapsed)
        gc.collect()
    result = {'runtime': 'python-mlx-adapted', 'batch': args.batch, 'codes': codes, 'ids': ids,
              'outputs_sha256_u32le': signature, 'warmups': 2 if args.bench else 0,
              'trials_seconds': timings, 'scope': 'generation and CPU FP64 output selection; load excluded',
              'mlx_memory': {'active': mx.get_active_memory(), 'peak': mx.get_peak_memory(), 'cache': mx.get_cache_memory()}}
    with args.output.open('x') as f:
        json.dump(result, f, sort_keys=True); f.write('\n')


if __name__ == '__main__':
    main()
