"""Unit tests for the offline `local-ncm-v1` exporter (stdlib only, no torch required)."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from export_local_ncm import (
    expected_codec_shapes,
    expected_generator_shapes,
    export_local_ncm,
    sha256_file,
    write_safetensors,
)


def make_config(head_kind: str = "untied_affine", vocab_sizes: list[int] | None = None) -> dict:
    vocabs = vocab_sizes or [12, 16, 20]
    return {
        "tokenizer": {
            "checkpoint_path": "local-codec.pt",
            "in_vocab_size": 32,
            "context_length": 8,
            "embed_dim": 16,
            "vqvae_vocab_size": vocabs,
            "n_layers": 2,
            "compression_factor": 1,
            "dropout": 0.0,
            "pre_quant_dropout": 0.0,
            "pre_quant_groupnorm": 4,
            "pre_quant_norm_type": "group",
            "use_qk_norm": True,
            "dec_layers": 2,
            "enc_positional": True,
            "dec_positional": False,
            "vector_quantizer_config": {
                "clss": "multiscale_residual_vector_quantizer",
                "commitment_cost": 0.25,
                "decay": 0.99,
                "epsilon": 1e-5,
                "learned_all_sampling": False,
                "learned_l1_sampling": True,
                "learned_l1_cascade": True,
                "learned_l1_cascade_base": 2,
                "learned_l1_cascade_norm": True,
                "learned_l1_cascade_norm_final": False,
                "levels": {
                    "use_manual_levels": True,
                    "manual_levels": [2, 4, 8],
                },
                "quant_resi": {
                    "enabled": True,
                    "learnable_ratio": False,
                    "ratio": 0.5,
                    "share_mode": 0,
                },
                "aux": {"fine_drop": {"min_keep": 1, "prob": 0.0}},
            },
        },
        "model": {
            "attn_pattern": "block_diagonal",
            "bias": True,
            "context_length": 24,
            "dropout": 0.0,
            "n_embd": 32,
            "n_head": 4,
            "n_layer": 2,
            "prefix_len": 6,
            "query_level0": True,
            "raw_token_prefix": true_bool(),
            "rope_base": 10000,
            "tie_heads_to_codebook": False,
            "use_level_encoding": True,
            "use_positional_encoding": True,
            "use_qk_norm": True,
            "use_rope": True,
            "vocab_size": vocabs,
            "head_kind": head_kind,
            "pad_token_id": 1,
            "eos_token_id": 0,
        },
    }


def true_bool() -> bool:
    return True


def make_zero_tensors(shapes: dict[str, list[int]]) -> dict[str, tuple[str, list[int], bytes]]:
    out = {}
    for name, shape in shapes.items():
        elems = 1
        for d in shape:
            elems *= d
        out[name] = ("F32", shape, struct.pack(f"<{elems}f", *([0.125] * elems)))
    return out


class TestExportLocalNcm(unittest.TestCase):
    def test_roundtrip_export_and_manifest_hashes(self) -> None:
        for head_kind, vocabs in (
            ("untied_affine", [12, 16, 20]),
            ("untied_linear", [12, 16, 20]),
            ("shared_head_biases", [16, 16, 16]),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = make_config(head_kind, vocabs)
                cfg_path = root / "config.json"
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

                codec_shapes = expected_codec_shapes(cfg["tokenizer"])
                gen_shapes = expected_generator_shapes(cfg["tokenizer"], cfg["model"])
                codec_tensors = make_zero_tensors(codec_shapes)
                # Include a training buffer to verify drop_training_buffers strips it cleanly.
                codec_tensors["vq.ema_cluster_size_0"] = (
                    "F32",
                    [vocabs[0]],
                    struct.pack(f"<{vocabs[0]}f", *([1.0] * vocabs[0])),
                )
                gen_tensors = make_zero_tensors(gen_shapes)

                codec_path = root / "codec.safetensors"
                gen_path = root / "gen.safetensors"
                write_safetensors(codec_path, codec_tensors)
                write_safetensors(gen_path, gen_tensors)

                out = export_local_ncm(
                    config_path=cfg_path,
                    codec_path=codec_path,
                    generator_path=gen_path,
                    output_dir=root / "artifact",
                )
                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["profile"], "local-ncm-v1")
                self.assertEqual(
                    set(manifest["files"]),
                    {"codec.safetensors", "generator.safetensors", "config.json", "provenance.json"},
                )
                for fname, rec in manifest["files"].items():
                    self.assertEqual(rec["sha256"], sha256_file(out / fname))
                    self.assertEqual(rec["bytes"], (out / fname).stat().st_size)

    def test_rejects_nonfinite_and_missing_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config()
            cfg_path = root / "config.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            codec_shapes = expected_codec_shapes(cfg["tokenizer"])
            gen_shapes = expected_generator_shapes(cfg["tokenizer"], cfg["model"])
            codec_tensors = make_zero_tensors(codec_shapes)
            gen_tensors = make_zero_tensors(gen_shapes)
            # Drop a required generator tensor.
            del gen_tensors["l0_queries"]
            codec_path = root / "codec.safetensors"
            gen_path = root / "gen.safetensors"
            write_safetensors(codec_path, codec_tensors)
            write_safetensors(gen_path, gen_tensors)
            with self.assertRaises(ValueError):
                export_local_ncm(cfg_path, codec_path, gen_path, root / "artifact")


if __name__ == "__main__":
    unittest.main()
