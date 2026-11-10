"""Generate pinned upstream Sudoku CUDA traces and qualify the exported roundtrip."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from sudoku_artifact import INPUTS, UPSTREAM, checkpoint, digest, json_file, tensor_bytes


def boards():
    solution = [[(row * 3 + row // 3 + col) % 9 + 1 for col in range(9)] for row in range(9)]
    targets, prefixes = [], []
    for shift in (0, 1):
        full, puzzle = [], []
        for row in range(9):
            if row:
                full.append(10)
                puzzle.append(10)
            for col in range(9):
                digit = (solution[row][col] + shift - 1) % 9 + 1
                full.append(digit)
                puzzle.append(digit if (row * 9 + col) % 3 else 0)
        targets.append(full)
        # Synthetic prefix format, not a claim about released evaluation-data packing.
        prefixes.append([11] + puzzle + [11])
    return torch.tensor(targets, device="cuda"), torch.tensor(prefixes, device="cuda")


def capture(codec, generator, target, prefix):
    traces = {}

    def save(name, tensor):
        traces[name] = tensor.detach().cpu().contiguous().clone()

    save("codec.input_ids", target)
    save("codec.pre_quant", codec.encode_z(target))
    codes = codec.encode(target)
    save("codec.codes", codes)
    save("codec.logits", codec.decode(codes))
    save("generator.prefix_ids", prefix)
    handles = []
    calls = [0]

    def before(_module, args):
        save(f"generator.level_{calls[0]:02d}.block0_input", args[0])

    def after(_module, _args, output):
        save(f"generator.level_{calls[0]:02d}.block0_output", output)
        calls[0] += 1

    handles.append(generator.transformer.blocks[0].register_forward_pre_hook(before))
    handles.append(generator.transformer.blocks[0].register_forward_hook(after))
    for level, head in enumerate(generator.lm_heads):
        def logits(_module, _args, output, level=level):
            save(f"generator.level_{level:02d}.logits", output)
        handles.append(head.register_forward_hook(logits))
    try:
        generated = generator.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0,
                                       use_kv_cache=False, cfg_scale=0.0)
    finally:
        for handle in handles:
            handle.remove()
    save("generator.codes", generated)
    z_hat = torch.zeros(target.shape[0], codec.vq.levels[-1], codec.embed_dim, device="cuda")
    offset = 0
    for level, length in enumerate(codec.vq.levels):
        q = codec.vq.embeddings[level](generated[:, offset:offset + length])
        z_hat, _ = codec.vq.get_next_autoregressive_input(level, z_hat, q)
        # Clone at capture time; never use the aliased idx_to_zhats z_hats return value.
        save(f"reconstruction.level_{level:02d}", z_hat)
        offset += length
    logits = codec.decoder(z_hat)
    save("generator.final_logits", logits)
    save("generator.output_ids", logits.argmax(-1))
    assert calls[0] == len(codec.vq.levels)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer-fixture", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.traces, args.report, args.tokenizer_fixture):
        if path.exists():
            parser.error(f"output exists: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no fallback")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    if revision != UPSTREAM or dirty:
        raise RuntimeError("upstream checkout must be clean and pinned")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE
    from tokenization.sudoku_tokenizer import SudokuTokenizer

    source_config = (args.source / "config.yaml").read_bytes()
    if digest(source_config) != INPUTS["config.yaml"]:
        raise RuntimeError("source config changed")
    config_data = json.loads((args.bundle / "config.json").read_text())
    if config_data != yaml.safe_load(source_config):
        raise RuntimeError("canonical config differs from pinned source")
    config = OmegaConf.create(config_data)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        data = (args.bundle / name).read_bytes()
        if digest(data) != record["sha256"] or len(data) != record["bytes"]:
            raise RuntimeError(f"artifact integrity mismatch: {name}")
    original = {}
    for name in ("tokenizer.pt", "generator.pt"):
        data = (args.source / name).read_bytes()
        if digest(data) != INPUTS[name]:
            raise RuntimeError("checkpoint source changed")
        original[name] = checkpoint(data)
    canonical_codec = load_file(args.bundle / "codec.safetensors")
    training = load_file(args.bundle / "training.safetensors")
    if canonical_codec.keys() & training.keys():
        raise RuntimeError("overlapping runtime/training tensor names")
    canonical_codec.update(training)
    canonical_generator = load_file(args.bundle / "generator.safetensors")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    target, prefix = boards()
    derived_buffers = {}

    def model_traces(codec_state, generator_state):
        codec = VQVAE(**config.tokenizer.vqvae)
        codec.load_state_dict(codec_state, strict=True)
        generator = NextConceptTransformer(codec, config.model)
        generator.load_state_dict(generator_state, strict=True)
        codec, generator = codec.cuda().eval(), generator.cuda().eval()
        for label, module in (("codec", codec), ("generator", generator)):
            persistent = module.state_dict().keys()
            derived_buffers[label] = {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "sha256": digest(tensor_bytes(tensor))}
                for name, tensor in module.named_buffers() if name not in persistent
            }
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            return capture(codec, generator, target, prefix)

    before = model_traces(original["tokenizer.pt"], original["generator.pt"])
    after = model_traces(canonical_codec, canonical_generator)
    if before.keys() != after.keys():
        raise RuntimeError("trace keys changed after conversion")
    hashes = {}
    for name in before:
        a, b = before[name], after[name]
        if a.shape != b.shape or a.dtype != b.dtype or tensor_bytes(a) != tensor_bytes(b):
            raise RuntimeError(f"conversion changed trace: {name}")
        if a.is_floating_point() and not torch.isfinite(a).all():
            raise RuntimeError(f"nonfinite model trace: {name}")
        hashes[name] = {"sha256": digest(tensor_bytes(a)), "shape": list(a.shape), "dtype": str(a.dtype)}
    save_file(before, args.traces)
    tokenizer = SudokuTokenizer()
    cases = []
    for text in ("", "<BOS> 0_12|9 B", "1\u001c\u00a0\u2003\n2", "123456789|_________", "\t<BOS>\r"):
        ids = tokenizer.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids),
                      "decoded_skip_special": tokenizer.decode(ids, skip_special_tokens=True)})
    json_file(args.tokenizer_fixture, {"schema_version": 1, "upstream_commit": UPSTREAM, "cases": cases})
    report = {"schema_version": 1, "upstream_commit": UPSTREAM, "torch_version": torch.__version__,
              "device": torch.cuda.get_device_name(0), "dtype": "float32", "tf32": False,
              "sdpa": "math", "kv_cache": False, "config_sha256": digest((args.bundle / "config.json").read_bytes()),
              "manifest_sha256": digest((args.bundle / "manifest.json").read_bytes()),
              "trace_script_sha256": digest(Path(__file__).read_bytes()),
              "artifact_script_sha256": digest(Path(__file__).with_name("sudoku_artifact.py").read_bytes()),
              "derived_buffers": derived_buffers,
              "traces_sha256": digest(args.traces.read_bytes()), "trace_count": len(hashes), "traces": hashes,
              "bit_exact_original_vs_converted": True, "examples": 2, "levels": len(config.tokenizer.vqvae.vqvae_vocab_size),
              "corpus_quality": None, "performance": None,
              "scope": "synthetic boards; upstream Python CUDA original vs canonical reload; not Rust model parity",
              "reference_patches": [], "snapshot_policy": "cloned captures avoid idx_to_zhats aliasing helper"}
    json_file(args.report, report)
    print(f"{len(hashes)} traces match bit-for-bit after canonical reload; tokenizer fixture exported")


if __name__ == "__main__":
    main()
