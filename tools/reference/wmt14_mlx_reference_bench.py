"""Independent WMT14 Python MLX adaptation for same-device parity checks."""
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx_codec import Codec, prepare_device
from mlx_codec_assets import tensors
from mlx_operators import execute
from phase3_mlx_generator import Generator

ARTIFACT_SHA = "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
PREFIX_SHA = "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f"


def checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected {path.name}")
    return raw


class WmtCodec(Codec):
    def up(self, x, level):
        if level == 0:
            for stage in range(3):
                if stage > 0:
                    x = execute({"kind": "group_norm", "groups": 8, "epsilon": 1e-5}, {"x": x})
                x = self.conv(f"vq.learned_up_l1.{stage * 2}", x, stride=4, transpose=True)
            return x
        return super().up(x, level)


class WmtGenerator(Generator):
    def __init__(self, root):
        manifest = json.loads(checked(root / "manifest.json", ARTIFACT_SHA))

        def payload(name):
            path = root / name
            spec = manifest["files"][name]
            if path.is_symlink() or path.stat().st_size != spec["bytes"]:
                raise ValueError(f"invalid {name}")
            return checked(path, spec["sha256"])

        config = json.loads(payload("config.json"))
        self.cfg = config["model"]
        codec_config = config["tokenizer"]["vqvae"]
        self.codec = WmtCodec(codec_config, {k: mx.array(x) for k, x in tensors(payload("codec.safetensors")).items()})
        self.levels = self.codec.levels
        self.w = {k: mx.array(x) for k, x in tensors(payload("generator.safetensors")).items()
                  if x.dtype == np.float32}
        self.geometry = []
        prefix = self.cfg["prefix_len"]
        half = self.cfg["n_embd"] // self.cfg["n_head"] // 2
        frequency = np.float32(self.cfg["rope_base"]) ** (np.arange(half, dtype=np.float32) / np.float32(half))
        for length in [0, *self.levels[1:]]:
            mask = np.zeros((1, 1, prefix + length, prefix + length), dtype=np.float32)
            mask[:, :, :prefix, prefix:] = -np.inf
            angles = np.concatenate((np.arange(prefix), np.arange(length))).astype(np.float32)[:, None] / frequency
            cosine, sine = np.cos(angles), np.sin(angles)
            self.geometry.append((mx.array(mask), mx.array(np.concatenate((cosine, cosine), axis=-1)),
                                  mx.array(np.concatenate((sine, sine), axis=-1))))
        mx.eval(*self.codec.w.values(), *self.w.values(), *(v for group in self.geometry for v in group))

    def hidden(self, x, padding, level):
        mask, cosine, sine = self.geometry[level]
        mask = mask + padding
        batch, seq, width = x.shape
        heads = self.cfg["n_head"]
        for i in range(self.cfg["n_layer"]):
            name = f"transformer.blocks.{i}"
            norm = self.norm(name + ".ln_1", x)
            qkv = self.linear(name + ".attn.c_attn", norm)
            q, k, v = [part.reshape(batch, seq, heads, -1).transpose(0, 2, 1, 3)
                       for part in mx.split(qkv, 3, axis=2)]
            q = self.norm(name + ".attn.q_norm", q, rms=True)
            k = self.norm(name + ".attn.k_norm", k, rms=True)
            q = execute({"kind": "rope"}, {"x": q, "cos": cosine, "sin": sine})
            k = execute({"kind": "rope"}, {"x": k, "cos": cosine, "sin": sine})
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask=mask)
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq, width)
            x = x + self.linear(name + ".attn.c_proj", attended)
            value = self.linear(name + ".mlp.c_fc", self.norm(name + ".ln_2", x))
            activated = 0.5 * value * (1 + mx.erf(value / math.sqrt(2)))
            x = x + self.linear(name + ".mlp.c_proj", activated)
        return self.norm("transformer.ln_f", x)

    def output(self, rows, scale):
        count = len(rows)
        if scale:
            null = np.full_like(rows, 151643)
            null[:, 0] = 151645
            rows = np.concatenate((rows, null))
        prefix_length = self.cfg["prefix_len"]
        ids = mx.array(rows, dtype=mx.uint32)
        positions = mx.array(np.count_nonzero(rows != 151643, axis=1) - 1)
        prefix = self.linear("transformer.wte", self.codec.w["encoder.wte.weight"][ids])
        selected, latent = [], None
        for level, length in enumerate(self.levels):
            mask = np.zeros((len(rows), 1, 1, prefix_length + (length if level else 0)), dtype=np.float32)
            mask[:, 0, 0, :prefix_length] = np.where(rows == 151643, -np.inf, 0.0)
            if level == 0:
                hidden = self.hidden(prefix, mx.array(mask), level)
                hidden = hidden[mx.arange(len(rows)), positions][:, None]
                head = "lm_head_l0.weight"
            else:
                target = execute({"kind": "area1d", "length": length}, {"x": latent})
                target = self.linear("transformer.wte", target) + self.w["transformer.wle.weight"][level - 1]
                hidden = self.hidden(mx.concatenate((prefix, target), axis=1), mx.array(mask), level)[:, prefix_length:]
                head = f"lm_heads.{level - 1}.weight"
            scores = hidden @ self.w[head].T
            if scale:
                t = scale * (level / (len(self.levels) - 1))
                scores = scores[:count] * mx.array(1 + t, dtype=mx.float32) - scores[count:] * mx.array(t, dtype=mx.float32)
            code = mx.argmax(scores, axis=-1)
            selected.append(code)
            both = mx.concatenate((code, code), axis=0) if scale else code
            book = self.codec.w[f"vq.embeddings.{level}.weight"]
            result = self.codec.phi(self.codec.up(book[both], level), level)
            latent = result if latent is None else latent + result
        codes = mx.concatenate(selected, axis=1)
        decoded = self.codec.norm("decoder.ln_f", self.codec.block(latent[:count]))
        ids = mx.argmax(self.codec.linear("decoder.output_layer", decoded), axis=-1)
        mx.eval(codes, ids)
        return {"codes": codes.tolist(), "raw_ids": ids.tolist()}


def main():
    artifact, prefix, output = map(Path, sys.argv[1:4])
    scale, start, batch = map(int, sys.argv[4:7])
    if scale not in (0, 1) or start < 0 or not 1 <= batch <= 32:
        raise ValueError("invalid WMT14 comparison range")
    warmups, trials = map(int, sys.argv[7:9]) if len(sys.argv) == 9 else (0, 1)
    if not 0 <= warmups <= 100 or not 1 <= trials <= 100:
        raise ValueError("invalid warmup or trial count")
    all_rows = json.loads(checked(prefix, PREFIX_SHA))["prefixes"]
    rows = np.asarray(all_rows[start:start + batch], dtype=np.uint32)
    if rows.shape != (batch, 96):
        raise ValueError("unexpected fixture dimensions")
    load_start = perf_counter()
    prepare_device()
    model = WmtGenerator(artifact)
    load_seconds = perf_counter() - load_start
    durations = []
    result = None
    for trial in range(warmups + trials):
        start_time = perf_counter()
        generated = model.output(rows, scale)
        elapsed = perf_counter() - start_time
        if result is not None and result != generated:
            raise ValueError("output changed between trials")
        result = generated
        if trial >= warmups:
            durations.append(elapsed)
    with output.open("x") as stream:
        json.dump({"scope": "adapted Python MLX WMT14, synchronized output readback", "batch": batch,
                   "case_start": start, "cfg": scale, "load_seconds": load_seconds,
                   "warmups": warmups, "trials_seconds": durations,
                   "mlx_memory": {"active": mx.get_active_memory(), "peak": mx.get_peak_memory(),
                                  "cache": mx.get_cache_memory()}, **result},
                  stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
