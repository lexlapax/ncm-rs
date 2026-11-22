#!/usr/bin/env python3
"""Offline exporter for generic config-driven `local-ncm-v1` artifacts.

Converts local NCM codec + generator weights (`.safetensors` with stdlib only, or
`.pt`/`.pth` via `torch.load(..., weights_only=True)`) plus a `config.json` and
optional HuggingFace `tokenizer.json` into a verified `local-ncm-v1` artifact
directory with SHA-256 file and tensor manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import struct
import tempfile
from pathlib import Path

SUPPORTED_DTYPES = {"F32": ("<f", 4), "I64": ("<q", 8)}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_safetensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated safetensors file: {path}")
    (header_len,) = struct.unpack("<Q", raw[:8])
    if header_len > 100_000_000 or 8 + header_len > len(raw):
        raise ValueError(f"invalid safetensors header length in {path}")
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r} for tensor {name!r}; expected F32 or I64")
        shape = [int(d) for d in info["shape"]]
        start, end = info["data_offsets"]
        payload = raw[data_start + start : data_start + end]
        _, elem_size = SUPPORTED_DTYPES[dtype]
        expected_bytes = elem_size
        for dim in shape:
            expected_bytes *= dim
        if len(payload) != expected_bytes:
            raise ValueError(f"tensor {name!r} byte length mismatch")
        tensors[name] = (dtype, shape, payload)
    return tensors


def load_weights(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    if path.suffix == ".safetensors":
        return read_safetensors(path)
    if path.suffix in (".pt", ".pth", ".bin"):
        import torch  # Lazy import: only needed for PyTorch checkpoints.

        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"expected tensor dict in {path}")
        out: dict[str, tuple[str, list[int], bytes]] = {}
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            t = tensor.detach().cpu().contiguous()
            if t.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
                t = t.to(torch.float32)
                dtype = "F32"
            elif t.dtype in (torch.int64, torch.int32):
                t = t.to(torch.int64)
                dtype = "I64"
            else:
                raise ValueError(f"unsupported torch dtype {t.dtype} for {name!r}")
            out[name] = (dtype, list(t.shape), t.numpy().tobytes())
        return out
    raise ValueError(f"unsupported weight format for {path} (expected .safetensors or .pt)")


def check_finite_f32(name: str, payload: bytes) -> None:
    if name == "attn_mask_for_masking":
        return
    if len(payload) % 4 != 0:
        raise ValueError(f"invalid F32 payload length for {name!r}")
    for (val,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(val):
            raise ValueError(f"non-finite F32 value in tensor {name!r}")


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> list[dict[str, object]]:
    header: dict[str, object] = {}
    records: list[dict[str, object]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        if dtype == "F32":
            check_finite_f32(name, payload)
        end = offset + len(payload)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, end],
        }
        records.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        chunks.append(payload)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for chunk in chunks:
            fh.write(chunk)
    return records


def expected_codec_shapes(tok: dict[str, object]) -> dict[str, list[int]]:
    d = int(tok["embed_dim"])
    vocab = int(tok["in_vocab_size"])
    ctx = int(tok["context_length"])
    heads = int(tok["n_layers"])
    dec_layers = int(tok.get("dec_layers", 1))
    enc_pos = bool(tok.get("enc_positional", True))
    dec_pos = bool(tok.get("dec_positional", False))
    vq_vocabs = [int(v) for v in tok["vqvae_vocab_size"]]
    q = tok["vector_quantizer_config"]
    levels = [int(v) for v in q["levels"]["manual_levels"]]
    first = levels[0]
    cascade = bool(q["learned_l1_cascade"])
    cascade_base = int(q["learned_l1_cascade_base"])
    cascade_norm = bool(q["learned_l1_cascade_norm"])

    remaining = ctx // first
    strides: list[int] = []
    if cascade:
        while remaining > 1:
            factor = min(cascade_base, remaining)
            while factor > 1 and remaining % factor != 0:
                factor -= 1
            if factor <= 1:
                factor = remaining
            strides.append(factor)
            remaining //= factor
    else:
        strides.append(remaining)

    shapes: dict[str, list[int]] = {
        "encoder.wte.weight": [vocab, d],
        "decoder.ln_f.weight": [d],
        "decoder.output_layer.weight": [vocab, d],
    }
    if enc_pos:
        shapes["encoder.wpe.weight"] = [ctx, d]
    if dec_pos:
        shapes["decoder.wpe.weight"] = [ctx, d]
    blocks = ["decoder.middle_block"] + [f"decoder.extra_blocks.{i}" for i in range(dec_layers - 1)]
    for block in blocks:
        for layer in ("ln_1", "ln_2"):
            for suffix in ("weight", "bias"):
                shapes[f"{block}.{layer}.{suffix}"] = [d]
        for layer in ("q_norm", "k_norm"):
            shapes[f"{block}.attn.{layer}.weight"] = [d // heads]
        for layer, out_d, in_d in (
            ("attn.c_attn", 3 * d, d),
            ("attn.c_proj", d, d),
            ("mlp.c_fc", 4 * d, d),
            ("mlp.c_proj", d, 4 * d),
        ):
            shapes[f"{block}.{layer}.weight"] = [out_d, in_d]
            shapes[f"{block}.{layer}.bias"] = [out_d]
    for i, size in enumerate(vq_vocabs):
        shapes[f"vq.embeddings.{i}.weight"] = [size, d]
        shapes[f"vq.quant_resi.{i}.weight"] = [d, d, 3]
        shapes[f"vq.quant_resi.{i}.bias"] = [d]
    for name, seq in (("down", strides), ("up", list(reversed(strides)))):
        for i, stride in enumerate(seq):
            if not cascade:
                shapes[f"vq.learned_{name}_l1.weight"] = [d, d, stride]
                shapes[f"vq.learned_{name}_l1.bias"] = [d]
            else:
                idx = 2 * i if cascade_norm else i
                shapes[f"vq.learned_{name}_l1.{idx}.weight"] = [d, d, stride]
                shapes[f"vq.learned_{name}_l1.{idx}.bias"] = [d]
    return shapes


def expected_generator_shapes(tok: dict[str, object], model: dict[str, object]) -> dict[str, list[int]]:
    d = int(model["n_embd"])
    heads = int(model["n_head"])
    layers = int(model["n_layer"])
    codec_d = int(tok["embed_dim"])
    levels = [int(v) for v in tok["vector_quantizer_config"]["levels"]["manual_levels"]]
    vocab_sizes = [int(v) for v in model["vocab_size"]]
    head_kind = model.get("head_kind", "untied_affine")

    shapes: dict[str, list[int]] = {
        "transformer.wte.weight": [d, codec_d],
        "transformer.wte.bias": [d],
        "l0_queries": [1, levels[0], codec_d],
        "transformer.wle.weight": [len(levels), d],
        "transformer.ln_f.weight": [d],
        "transformer.ln_f.bias": [d],
    }
    if head_kind == "untied_affine":
        for i, v in enumerate(vocab_sizes):
            shapes[f"lm_heads.{i}.weight"] = [v, d]
            shapes[f"lm_heads.{i}.bias"] = [v]
    elif head_kind == "untied_linear":
        for i, v in enumerate(vocab_sizes):
            shapes[f"lm_heads.{i}.weight"] = [v, d]
    elif head_kind == "shared_head_biases":
        shapes["lm_head.weight"] = [vocab_sizes[0], d]
        for i, v in enumerate(vocab_sizes):
            shapes[f"head_biases.{i}"] = [v]
    else:
        raise ValueError(f"unsupported head_kind: {head_kind!r}")

    for i in range(layers):
        p = f"transformer.blocks.{i}"
        for layer in ("ln_1", "ln_2"):
            shapes[f"{p}.{layer}.weight"] = [d]
            shapes[f"{p}.{layer}.bias"] = [d]
        for qk in ("q_norm", "k_norm"):
            shapes[f"{p}.attn.{qk}.weight"] = [d // heads]
        for layer, out_d, in_d in (
            ("attn.c_attn", 3 * d, d),
            ("attn.c_proj", d, d),
            ("mlp.c_fc", 4 * d, d),
            ("mlp.c_proj", d, 4 * d),
        ):
            shapes[f"{p}.{layer}.weight"] = [out_d, in_d]
            shapes[f"{p}.{layer}.bias"] = [out_d]
    return shapes


def validate_tensors(
    bundle_name: str,
    loaded: dict[str, tuple[str, list[int], bytes]],
    expected: dict[str, list[int]],
    optional_names: set[str] | None = None,
) -> dict[str, tuple[str, list[int], bytes]]:
    optional_names = optional_names or set()
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise ValueError(f"{bundle_name} missing required tensors: {missing[:5]}")
    extra = sorted(set(loaded) - set(expected) - optional_names)
    if extra:
        raise ValueError(f"{bundle_name} contains unexpected tensors: {extra[:5]}")
    out: dict[str, tuple[str, list[int], bytes]] = {}
    for name, exp_shape in expected.items():
        dtype, shape, payload = loaded[name]
        if dtype != "F32" or shape != exp_shape:
            raise ValueError(
                f"{bundle_name}/{name}: expected F32 {exp_shape}, got {dtype} {shape}"
            )
        out[name] = (dtype, shape, payload)
    for name in sorted(optional_names):
        if name in loaded:
            out[name] = loaded[name]
    return out


def export_local_ncm(
    config_path: Path,
    codec_path: Path,
    generator_path: Path,
    output_dir: Path,
    tokenizer_path: Path | None = None,
    source_repo: str = "local/custom-ncm",
    source_revision: str = "local-v1",
    upstream_commit: str = "local-export",
    drop_training_buffers: bool = True,
) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output path already exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != {"tokenizer", "model"}:
        raise ValueError("config.json must have exactly top-level keys 'tokenizer' and 'model'")

    codec_loaded = load_weights(codec_path)
    if drop_training_buffers:
        codec_loaded = {
            k: v
            for k, v in codec_loaded.items()
            if not k.startswith(("vq.ema_cluster_size_", "vq.ema_w_", "vq.codebook_hit_"))
        }
    gen_loaded = load_weights(generator_path)

    codec_tensors = validate_tensors(
        "codec.safetensors", codec_loaded, expected_codec_shapes(config["tokenizer"])
    )
    gen_tensors = validate_tensors(
        "generator.safetensors",
        gen_loaded,
        expected_generator_shapes(config["tokenizer"], config["model"]),
        optional_names={"lvl_1L", "attn_mask_for_masking"},
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".ncm-local-export-", dir=output_dir.parent))
    try:
        tensor_records: dict[str, list[dict[str, object]]] = {}
        tensor_records["codec.safetensors"] = write_safetensors(
            tmp_dir / "codec.safetensors", codec_tensors
        )
        tensor_records["generator.safetensors"] = write_safetensors(
            tmp_dir / "generator.safetensors", gen_tensors
        )
        (tmp_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if tokenizer_path is not None:
            shutil.copyfile(tokenizer_path, tmp_dir / "tokenizer.json")
        provenance = {
            "exporter": "tools/reference/export_local_ncm.py",
            "profile": "local-ncm-v1",
            "python": platform.python_version(),
            "source_files": {
                "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
                "codec": {"path": codec_path.name, "sha256": sha256_file(codec_path)},
                "generator": {"path": generator_path.name, "sha256": sha256_file(generator_path)},
            },
        }
        if tokenizer_path is not None:
            provenance["source_files"]["tokenizer"] = {
                "path": tokenizer_path.name,
                "sha256": sha256_file(tokenizer_path),
            }
        (tmp_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        file_records: dict[str, dict[str, object]] = {}
        for item in sorted(tmp_dir.iterdir()):
            file_records[item.name] = {
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        manifest = {
            "schema_version": 1,
            "profile": "local-ncm-v1",
            "upstream_commit": upstream_commit,
            "checkpoint_repository": source_repo,
            "checkpoint_revision": source_revision,
            "files": file_records,
            "tensors": tensor_records,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local NCM config.json")
    parser.add_argument(
        "--codec", type=Path, required=True, help="Codec weights (.safetensors or .pt)"
    )
    parser.add_argument(
        "--generator", type=Path, required=True, help="Generator weights (.safetensors or .pt)"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output artifact directory")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional HF tokenizer.json")
    parser.add_argument("--source-repo", default="local/custom-ncm")
    parser.add_argument("--source-revision", default="local-v1")
    parser.add_argument("--upstream-commit", default="local-export")
    args = parser.parse_args()
    out = export_local_ncm(
        config_path=args.config,
        codec_path=args.codec,
        generator_path=args.generator,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        upstream_commit=args.upstream_commit,
    )
    print(f"Exported local-ncm-v1 artifact to {out}")


if __name__ == "__main__":
    main()
