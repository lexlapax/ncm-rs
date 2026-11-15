"""Pinned Python Sudoku reference for comparison; offline diagnostics only."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from sudoku_artifact import INPUTS, INPUT_SIZES, PHASE3_INPUTS, PHASE3_SIZES, UPSTREAM, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--difficulty", choices=("easy", "medium", "hard"), required=True)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--capture-logits", action="store_true")
    p.add_argument("--bench", action="store_true", help="five warmups and 30 synchronized trials of one batch")
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.batch_size <= 128:
        raise ValueError("output must be new and batch size in 1..=128")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise ValueError("upstream revision mismatch")
    if subprocess.check_output(["git", "-C", str(a.upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("upstream source checkout is dirty")
    hashes = INPUTS if a.difficulty == "easy" else PHASE3_INPUTS[a.difficulty]
    sizes = INPUT_SIZES if a.difficulty == "easy" else PHASE3_SIZES[a.difficulty]
    data = {}
    for name, expected in hashes.items():
        path = a.source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != sizes[name]:
            raise ValueError("source file type or size differs")
        data[name] = path.read_bytes()
        if hashlib.sha256(data[name]).hexdigest() != expected:
            raise ValueError("source SHA-256 differs")
    cfg = OmegaConf.create(yaml.safe_load(data["config.yaml"]))
    if cfg.task != f"sudoku-{a.difficulty}": raise ValueError("wrong task")
    rows = json.loads(a.inputs.read_bytes())
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2000 or any(not isinstance(row, list) or len(row) != cfg.model.prefix_len or any(type(x) is not int or not 0 <= x < cfg.tokenizer.vqvae.in_vocab_size for x in row) for row in rows):
        raise ValueError("invalid Sudoku prefix inputs")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(checkpoint(data["tokenizer.pt"]), strict=True)
    model = NextConceptTransformer(codec, cfg.model)
    model.load_state_dict(checkpoint(data["generator.pt"]), strict=True)
    model, codec = model.cuda().eval(), codec.cuda().eval()
    del data  # Source bytes are verified before loading and need not remain resident.
    if a.bench:
        if a.capture_logits or len(rows) != a.batch_size:
            raise ValueError("benchmark requires one batch and no logit capture")
        prefix_cpu = torch.tensor(rows, dtype=torch.long, device="cpu")
        samples, expected = [], None
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for trial in range(35):
                torch.cuda.synchronize()
                start = time.perf_counter()
                prefix = prefix_cpu.to("cuda")
                code = model.generate(None, prefix, temperature=0, top_k=0, top_p=1, use_kv_cache=False, cfg_scale=0)
                ids = codec.decode(code).argmax(-1)
                cpu_codes, cpu_ids = code.cpu(), ids.cpu()
                elapsed = time.perf_counter() - start
                digest = hashlib.sha256(cpu_codes.numpy().tobytes() + cpu_ids.numpy().tobytes()).hexdigest()
                if expected is not None and digest != expected: raise ValueError("outputs changed between trials")
                expected = digest
                if trial >= 5: samples.append(elapsed)
        report = {"batch": len(rows), "warmups": 5, "trials": samples, "outputs_sha256_i64le": expected,
                  "scope": "generate plus code and final-ID host readback; excludes model load",
                  "torch_allocated_peak": torch.cuda.max_memory_allocated(),
                  "torch_reserved_peak": torch.cuda.max_memory_reserved(), "sdpa": "math", "tf32": False,
                  "source_sha256": hashes, "upstream": UPSTREAM, "torch": torch.__version__}
        with a.output.open("x") as file: json.dump(report, file, separators=(",", ":")); file.write("\n")
        return
    codes, outputs, logits = [], [], {}
    handles = []
    if a.capture_logits:
        if len(rows) > a.batch_size: raise ValueError("logit capture requires one batch")
        for i, head in enumerate(model.lm_heads):
            def save(_module, _args, output, index=i):
                logits[f"generator.level_{index:02}.logits"] = output.detach().cpu().flatten().tolist()
            handles.append(head.register_forward_hook(save))
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for offset in range(0, len(rows), a.batch_size):
            prefix = torch.tensor(rows[offset:offset + a.batch_size], dtype=torch.long, device="cuda")
            code = model.generate(None, prefix, temperature=0, top_k=0, top_p=1, use_kv_cache=False, cfg_scale=0)
            ids = codec.decode(code).argmax(-1)
            codes.extend(code.cpu().tolist())
            outputs.extend(ids.cpu().tolist())
    torch.cuda.synchronize()
    for handle in handles: handle.remove()
    report = {"logits": logits, "codes": codes, "ids": outputs, "inputs_sha256": hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
              "difficulty": a.difficulty, "upstream": UPSTREAM, "torch": torch.__version__,
              "cuda_device": torch.cuda.get_device_name(0), "batch_size": a.batch_size,
              "elapsed_seconds": time.perf_counter() - start, "sdpa": "math", "tf32": False,
              "cache": False, "source_sha256": hashes}
    with a.output.open("x") as file: json.dump(report, file, separators=(",", ":")); file.write("\n")

if __name__ == "__main__": main()
