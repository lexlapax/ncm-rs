"""Run a pinned LM1B-small request with the Python MLX development reference."""
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_codec import Codec, prepare_device
from mlx_codec_assets import digest, tensors
from mlx_operators import execute
from phase3_mlx_generator import Generator


MANIFEST_SHA = '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76'
ROTARY_SHA = '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc'
WORD_MASK = (1 << 64) - 1


def verified_bytes(path, size, sha):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f'invalid {path.name} file')
    raw = path.read_bytes()
    if len(raw) != size or digest(raw) != sha:
        raise ValueError(f'{path.name} differs from the pinned artifact')
    return raw


class ControlledDraws:
    def __init__(self, seed):
        self.state = seed

    def choose(self, weights):
        self.state = (self.state + 0x9e3779b97f4a7c15) & WORD_MASK
        mixed = self.state
        mixed = ((mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9) & WORD_MASK
        mixed = ((mixed ^ (mixed >> 27)) * 0x94d049bb133111eb) & WORD_MASK
        mixed ^= mixed >> 31
        threshold = ((mixed >> 32) + 0.5) / (1 << 32) * sum(weights)
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative > threshold:
                return index
        raise ValueError('invalid categorical distribution')

    def from_logits(self, logits):
        peak = max(logits)
        return self.choose([math.exp(value - peak) for value in logits])


class Lm1bCodec(Codec):
    def up(self, x, level):
        if level == 0:
            return self.conv('vq.learned_up_l1', x, stride=self.length, transpose=True)
        return super().up(x, level)

    def logits(self, cumulative):
        x = self.block(cumulative + self.w['decoder.wpe.weight'])
        for index in range(self.config['dec_layers'] - 1):
            x = self.block(x, prefix=f'decoder.extra_blocks.{index}')
        return self.linear('decoder.output_layer', self.norm('decoder.ln_f', x))


class Lm1bGenerator(Generator):
    def __init__(self, artifact, rotary):
        if artifact.is_symlink() or not artifact.is_dir():
            raise ValueError('invalid LM1B artifact directory')
        manifest = json.loads(verified_bytes(artifact / 'manifest.json', 89_870, MANIFEST_SHA))
        tables = tensors(verified_bytes(rotary, 32_920, ROTARY_SHA))

        def payload(name):
            record = manifest['files'][name]
            return verified_bytes(artifact / name, record['bytes'], record['sha256'])

        config = json.loads(payload('config.json'))
        self.cfg = config['model']
        codec_config = config['tokenizer']['vqvae']
        self.levels = codec_config['vector_quantizer_config']['levels']['manual_levels']
        codec = tensors(payload('codec.safetensors'))
        generator = tensors(payload('generator.safetensors'))
        inference = tensors(payload('inference.safetensors'))
        self.initial_weights = inference['level0_unigram'].tolist()
        self.token_weights = inference['token_unigram'].astype(np.float64) * 128

        prepare_device()
        self.codec = Lm1bCodec(codec_config, {name: mx.array(value) for name, value in codec.items()})
        self.w = {name: mx.array(value) for name, value in generator.items() if value.dtype == np.float32}
        self.geometry = []
        for length in self.levels:
            mask = np.zeros((1, 1, length + 1, length + 1), dtype=np.float32)
            mask[:, :, :1, 1:] = -np.inf
            positions = np.concatenate(([0], np.arange(length)))
            cosine, sine = (tables[name][positions] for name in ('cosine', 'sine'))
            self.geometry.append((mx.array(mask), mx.array(np.tile(cosine, (1, 2))),
                                  mx.array(np.tile(sine, (1, 2)))))
        mx.eval(*self.codec.w.values(), *self.w.values(), *(v for group in self.geometry for v in group))

    def linear(self, name, x):
        weight, bias = self.w[name + '.weight'], self.w[name + '.bias']
        output = mx.addmm(bias, x.reshape(-1, x.shape[-1]), weight.T)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    def norm(self, name, x, rms=False):
        if rms:
            return mx.fast.rms_norm(x, None, 1e-6) * self.w[name + '.weight']
        return super().norm(name, x)

    def attention(self, q, k, v, mask):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask=mask)

    def sample(self, batch, seed):
        rng = ControlledDraws(seed)
        prefix_ids = mx.full((batch, 1), 101, dtype=mx.uint32)
        prefix = self.linear('transformer.wte', self.codec.w['encoder.wte.weight'][prefix_ids])
        initial = mx.array([[rng.choose(self.initial_weights)] for _ in range(batch)], dtype=mx.uint32)
        codes = []
        cumulative = None
        for level, length in enumerate(self.levels):
            if level == 0:
                chosen = initial
            else:
                latent = execute({'kind': 'area1d', 'length': length}, {'x': cumulative})
                conv = f'post_upsample_convs.{level}'
                delta = mx.conv1d(latent, self.w[conv + '.weight'].transpose(0, 2, 1), padding=1)
                target = self.linear('transformer.wte', latent + (delta + self.w[conv + '.bias']))
                hidden = self.level(prefix, target + self.w['transformer.wle.weight'][level], level)
                logits = hidden @ self.w['lm_heads.head.weight'].T + self.w[f'lm_heads.biases.{level}']
                rows = np.asarray(logits).reshape(batch * length, 8192)
                if not np.isfinite(rows).all():
                    raise ValueError('nonfinite generator logits')
                chosen = mx.array([rng.from_logits(row.tolist()) for row in rows], dtype=mx.uint32)
                chosen = chosen.reshape(batch, length)
            codes.append(chosen)
            upsampled = self.codec.up(self.codec.w[f'vq.embeddings.{level}.weight'][chosen], level)
            contribution = self.codec.phi(upsampled, level)
            cumulative = contribution if cumulative is None else cumulative + contribution
        logits = np.asarray(self.codec.logits(cumulative))
        if logits.shape != (batch, 128, 30528) or not np.isfinite(logits).all():
            raise ValueError('invalid decoder logits')
        ids = [self.penalized_ids(row) for row in logits]
        return {'codes': mx.concatenate(codes, axis=1).tolist(), 'ids': ids}

    def penalized_ids(self, logits):
        counts = np.zeros_like(self.token_weights)
        tokens = []
        for pos, row in enumerate(logits):
            scores = row.astype(np.float64)
            scores -= 11.5 * np.maximum(counts - self.token_weights, 0.)
            scores += 2. * ((128 - pos) / 128) * np.maximum(self.token_weights - counts, 0.)
            for token, penalty in zip(reversed(tokens), (10., 5., 2.5, 1.25)):
                scores[token] -= penalty
            token = int(np.argmax(scores))
            counts[token] += 1
            tokens.append(token)
        return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('rotary', type=Path)
    parser.add_argument('batch', type=int)
    parser.add_argument('seed', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.batch <= 8 or not 0 <= args.seed <= WORD_MASK:
        parser.error('batch must be in 1..8 and seed must fit in an unsigned 64-bit integer')
    if args.output.exists() or args.output.is_symlink():
        parser.error('output must be a new file')
    result = Lm1bGenerator(args.artifact, args.rotary).sample(args.batch, args.seed)
    with args.output.open('x') as file:
        json.dump(result, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
