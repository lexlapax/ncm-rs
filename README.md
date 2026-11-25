# ncm-rs

**Fast, zero-Python Rust runtime, OpenAI/Ollama server, and LoRA trainer for [Next-Concept Modeling (NCM)](https://github.com/HazyResearch/ncm) on Linux/CUDA and Apple Silicon/Metal.**

[![Release v0.7.0](https://img.shields.io/badge/release-v0.7.0-blue.svg)](https://github.com/lexlapax/ncm-rs/releases/tag/v0.7.0)
[![Paper arXiv:2610.14162](https://img.shields.io/badge/paper-arXiv%3A2610.14162-b31b1b.svg)](https://arxiv.org/abs/2610.14162)
[![Upstream HazyResearch/ncm](https://img.shields.io/badge/upstream-HazyResearch%2Fncm-black.svg)](https://github.com/HazyResearch/ncm)
[![HF Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97-hazyresearch%2Fncm-FFD21E.svg)](https://huggingface.co/collections/hazyresearch/ncm)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is Next-Concept Modeling (NCM)?

[Next-Concept Modeling](https://arxiv.org/abs/2610.14162) (HazyResearch, Stanford) is a two-stage framework for **parallel, coarse-to-fine text generation**:

1. **Multiscale Residual VQ Tokenizer (Codec):** compresses a token sequence into a hierarchy of discrete concept codes across progressively finer resolutions.
2. **Coarse-to-Fine Generator:** a transformer predicts the code hierarchy level by level. Every position within a level is generated **in parallel**, so producing an entire sequence takes one forward evaluation per level (e.g., 9–12 passes) instead of one per token — achieving strong quality at **9–31×** the speed of diffusion baselines in the upstream paper.

## Why `ncm-rs`?

Upstream `HazyResearch/ncm` is a PyTorch research codebase targeting Linux CUDA evaluation scripts. **`ncm-rs`** brings NCM to everyday local execution as a single verified Rust binary with **zero Python required in production**:

- **Zero-Python End-to-End Workflow:** `ncm pull`, `ncm quantize`, `ncm sample`, `ncm run`, `ncm serve`, `ncm train`, and `ncm doctor` all run natively in Rust on **Linux/CUDA (cuDNN)** and **Apple Silicon/Metal (MLX)**.
- **Verified Upstream Parity + Fast Low-Precision Modes:** default `--precision fp32` preserves strict parity against pinned upstream checkpoints on both GPUs; opt-in `bf16`, `fp16`, `q8`, and `q4` unlock custom SM80+ CUDA Tensor-Core FlashAttention-2, Metal Fast SDPA, native `Q8_0`/`Q4_0` & group-64 affine quantization, and pre-quantized SHA-256-verified sidecars (`ncm quantize`).
- **Drop-In OpenAI & Ollama Local Server:** zero-dependency HTTP/1.1 server (`ncm serve`) with SSE/NDJSON streaming, dynamic micro-batching (`--max-batch`), bounded LRU VRAM management, and **zero-reload multi-LoRA adapter hot-swapping** over a single shared quantized GPU base.
- **Native Rust Fine-Tuning & LoRA:** train custom `local-ncm-v1` codecs/generators from scratch or LoRA fine-tune upstream TinyStories, WMT14, XSum, and LM1B checkpoints directly on CUDA or Apple Silicon — exporting compact `~4.5 MiB` verified adapters (`ncm-lora-adapter-v1`) or standalone merged checkpoints.
- **Fail-Closed Verification:** every artifact, pre-quantized sidecar, and LoRA adapter is SHA-256 verified at load time; unsupported features fail explicitly rather than silently falling back.

---

## Quick Start

### 1. Install `ncm`

**Prebuilt release bundles** are published on [GitHub Releases (`v0.7.0`)](https://github.com/lexlapax/ncm-rs/releases/tag/v0.7.0) with `.sha256` checksums and a rendered Homebrew formula (`ncm.rb` for Apple Silicon):

- **Apple Silicon macOS (self-contained Metal/MLX bundle — no Xcode, Rust, CMake, or Python required):**
  ```sh
  tar -xzf ncm-v0.7.0-macos-arm64-mlx.tar.gz
  ./tools/install.sh --archive ncm-v0.7.0-macos-arm64-mlx.tar.gz
  # Or copy bin/ncm -> ~/.local/bin/ and lib/* -> ~/.local/lib/
  ```
- **Linux x86_64 (NVIDIA CUDA + cuDNN 9):**
  ```sh
  ./tools/install.sh --archive ncm-v0.7.0-linux-x86_64-cuda.tar.gz
  ```
- **Build from source:** see the [5-Minute Quickstart](docs/QUICKSTART.md#1-build-ncm) (includes one-command `./tools/setup-mlx-native.sh` on macOS).

### 2. Verify Your GPU, Pull a Model & Run

```sh
# Check GPU/driver/runtime readiness and run a fast zero-download kernel self-test:
ncm doctor --self-test

# Pull an upstream checkpoint natively in Rust (zero Python; SHA-256 verified):
ncm pull --list
ncm pull wmt14

# Run one-shot or interactive REPL prompts:
ncm run wmt14 "Die Forschung zeigt deutliche Fortschritte bei lokalen KI-Modellen." --precision fp16
ncm run wmt14 --precision q8

# Pre-quantize once for instant Q8/Q4 cold loads (3-5x smaller, 4.8-9.2x faster load):
ncm quantize "$(ncm list --json | jq -r '.[] | select(.name=="wmt14") | .path')" --precision q8 --device cuda
```

### 3. Serve Locally (OpenAI & Ollama Compatible)

```sh
# Start localhost server on 127.0.0.1:11434 (compatible with OpenAI SDK & Ollama clients):
ncm serve --precision fp16 --preload wmt14 --max-batch 8

# Hot-swap multiple LoRA adapters over ONE shared Q8 GPU base (zero base reloads):
ncm serve --precision q8 --max-loaded 1 --preload tinystories \
  --adapter pirate=tinystories:./adapters/pirate \
  --adapter scifi=tinystories:./adapters/scifi
```

### 4. Fine-Tune with Native Rust LoRA

```sh
# Export a verified ~4.5 MiB unmerged LoRA adapter on CUDA or Metal (TinyStories, WMT14, XSum, LM1B):
ncm train --init-from /path/to/tinystories-15k-ema-v1 \
  --data train.jsonl --val-data val.jsonl --eval-every 10 \
  --save-adapter ./adapters/pirate \
  --device cuda --lora-rank 8 --lora-alpha 16 --train-precision bf16 \
  --steps 50 --batch 2 --grad-accum 2 --lr 3e-4 --overwrite
```

> Globally installed binaries store models in `~/.local/share/ncm/models` (Linux) or `~/Library/Application Support/ncm/models` (macOS) by default, and `./artifacts/canonical` inside a repo checkout. Override anytime via `NCM_MODELS_DIR` or `--models-dir`.

---

## Performance Highlights

Measured on real target hardware (**Linux:** NVIDIA RTX 3060 Ti 8 GiB, SM86; **macOS:** Apple M1 Ultra 64 GiB, native MLX). Every number links to reproducible commands, quality checks, and raw JSON receipts.

| Area | Linux / CUDA (RTX 3060 Ti) | Apple Silicon / Metal (M1 Ultra) | Detailed Report |
| --- | --- | --- | --- |
| **Warm latency vs upstream Python (`tinystories-15k-ema`, b=1)** | Eager Rust `bf16` **26.67 ms / 253.6 MiB** beats Python `094f25ca` `torch.compile`+CUDA graphs (`31.28 ms / 647.9 MiB`) by **1.17×** at **2.55× less VRAM** | `bf16` **76.54 ms (`q8`)** / **78.58 ms (`bf16`)**, **1.22×–1.72×** faster than FP32 baseline | [Phase 8 & 9](docs/results/2026-11-23-phase8-phase9/README.md) · [Phase 10B](docs/results/2026-11-24-phase10/README.md) |
| **Fused Attention (FlashAttention-2 / Metal Fast SDPA)** | `xsum` b=8 `bf16` **1.070× faster (`-36.0 MiB` VRAM)**; `tinystories` b=8 **1.042×** | `xsum` b=1 `bf16` **1.056× faster (`-63.1 MiB` MLX peak)**; `wmt14` b=8 **-29.8 MiB** | [Phase 10 Report](docs/results/2026-11-24-phase10/README.md) |
| **`Q8`/`Q4` Weight Quantization (`lm1b-large`, 860M)** | b=1 VRAM **-68.1% (`q8`, 1,025 MiB)** / **-79.1% (`q4`, 673 MiB)**; **b=16 fits in 975 MiB** | b=1 MLX peak **-66.9% (`q8`) / -77.2% (`q4`, 780 MiB)**; RSS **-90.2%** | [Phase 7 Report](docs/results/2026-11-23-phase7/README.md) |
| **Pre-Quantized Sidecar Cold Loads (`ncm quantize`)** | **4.79×–9.23× faster** (`lm1b-large` `q4`: `3,861 → 418.5 ms`, **5.45× smaller**) | **2.50×–4.75× faster** (`lm1b-large` `q4`: `2,436 → 512.5 ms`, **5.45× smaller**) | [Phase 8B Report](docs/results/2026-11-23-phase8-phase9/README.md) |
| **Concurrent Serving (`--max-batch 8`) & Multi-LoRA Swap** | **2.21×** on `wmt14` (`3,560.8 tok/s`), **1.64×** on `lm1b-large` Q8; adapter hot-swap **`load_ns = 0`** (`44.8 ms` base / `48.0 ms` adapter) | **1.85×** on `wmt14`, **1.97×** on `lm1b-large` Q8; adapter hot-swap **`load_ns = 0`** (`81.7 ms` base / `91.8 ms` adapter) | [Phase 7](docs/results/2026-11-23-phase7/README.md) · [Phase 10C](docs/results/2026-11-24-phase10/README.md) |
| **Native Rust LoRA (`--save-adapter`, rank 8)** | All 4 upstream profiles converge in **< 3.7 GiB free VRAM** (`4.53 MiB` adapter = **121.7×–240.0×** smaller) | All 4 upstream profiles converge at **304–641 ms/step** (`4.53 MiB` adapter, 4-decimal parity vs CUDA) | [Phase 10C Report](docs/results/2026-11-24-phase10/README.md) |

---

## Supported Models & Upstream Parity

All released checkpoints from [hazyresearch/ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints) are supported and dual-platform verified (plus user-trained `local-ncm-v1` models):

| Model Family | `ncm pull` Targets | Qualified Modes | Parity & Quality Evidence |
| --- | --- | --- | --- |
| **WMT14 De–En** | `wmt14` | Top-1 (`CFG 0`/`1`), `fp32`/`bf16`/`fp16`/`q8`/`q4`, serving, LoRA | [WMT14 qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification) · [Prefix KV & precision](docs/results/2026-11-22-phase6-precision-serving/README.md) |
| **XSum** | `xsum` | Top-1 (`CFG 0`), `fp32`/`bf16`/`fp16`/`q8`/`q4`, serving, LoRA (seeded/CFG opt-in) | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition) · [Fused attention](docs/results/2026-11-24-phase10/README.md) |
| **TinyStories** | `tinystories` (15k EMA), `tinystories-legacy` | Seeded, `fp32`/`bf16`/`q8`/`q4` (`fp16` explicitly rejected on `4.37e10` outliers), serving, LoRA | [Upstream `094f25ca` qualification](docs/results/2026-11-23-upstream-qualification/README.md) (`0/2,564` code diffs) |
| **LM1B (Small / Medium / Large)** | `lm1b-small`, `lm1b-medium`, `lm1b-large` | Seeded (`v1` + self-contained `v2`), `fp32`/`bf16`/`fp16`/`q8`/`q4`, serving, LoRA | [LM1B cohort campaign](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md) · [vs Python MLX](docs/results/2026-11-20-phase5-optimization/README.md) |
| **Sudoku (Easy / Medium / Hard)** | `sudoku-easy`, `sudoku-medium`, `sudoku-hard` | Greedy `fp32`/`bf16`/`fp16`/`q8`/`q4` | [6,000-board quality & timing](docs/results/2026-11-14-phase3-sudoku/README.md) (`128/128` exact in `bf16`/`fp16`/`q8`) |
| **Custom Local NCM** | `local-ncm-v1` (via `ncm train` or `export_local_ncm.py`) | Greedy & seeded, `fp32`/`bf16`/`fp16`/`q8`/`q4`, serving, CPU/CUDA/Metal | [Developer Guide](docs/DEVELOPING.md#4-running-your-own-local-ncm-model-local-ncm-v1) · [Artifact Contract](docs/ARTIFACTS.md) |
| **OpenWebText (OWT)** | *(unreleased upstream)* | Config & synthetic operator coverage only; inference refused until weights publish | [OWT coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md) |

---

## Guides & Documentation by Theme

### User Guides & Quickstarts
- **[5-Minute Quickstart](docs/QUICKSTART.md)** — fastest path from install/build to `ncm doctor`, `ncm pull`, `ncm run`, OpenAI/Ollama `ncm serve`, `ncm quantize`, and LoRA `ncm train`.
- **[Full CLI & Reproducible Usage Guide](docs/USAGE.md)** — every command, experimental flag boundary, exact-hash artifact reproduction, and fresh-clone guides ([Linux](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md), [macOS](docs/results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md)).
- **[Artifact & Adapter Contract](docs/ARTIFACTS.md)** — `manifest.json`, `local-ncm-v1`, pre-quantized sidecar, and `ncm-lora-adapter-v1` integrity rules.

### Performance & Benchmarking Guides
- **[Published Benchmark & Comparison Index](docs/results/README.md)** — master index of all dual-platform quality, latency, throughput, and memory reports.
- **[Low-Precision Kernels, Sidecars & LoRA Performance (Phase 8 & 9)](docs/results/2026-11-23-phase8-phase9/README.md)** — native `bf16`/`fp16`/`q8`/`q4` kernels, Rust vs upstream Python `torch.compile`+CUDA graphs, and `ncm quantize` load speedups.
- **[Fused Attention & Multi-Adapter Serving (Phase 10)](docs/results/2026-11-24-phase10/README.md)** — CUDA SM80+ FlashAttention-2, Metal Fast SDPA, zero-Python `ncm pull`, and multi-LoRA Q8 serving.
- **[Quantization, Micro-Batching & Native Training (Phase 7)](docs/results/2026-11-23-phase7/README.md)** — `Q8`/`Q4` VRAM & accuracy tradeoffs, 8-client serving throughput, and native Rust training.
- **[Stage-Breakdown Profiling & Load Optimization (Phase 5)](docs/results/2026-11-20-phase5-profile/README.md)** — per-stage bottleneck attribution, [prefix K/V caching & streaming loader wins](docs/results/2026-11-20-phase5-optimization/README.md).
- **[Packaging & Relocatable Bundles (Phase 11)](docs/results/2026-11-24-phase11/README.md)** — Linux/CUDA & self-contained macOS/Metal bundle verification and `ncm doctor --self-test` telemetry.

### Architecture, Development & Maintenance
- **[Architecture](docs/ARCHITECTURE.md)** & **[Developer Guide](docs/DEVELOPING.md)** — workspace boundaries, backend traits (`CodecBackend`, `GeneratorBackend`), adding custom models, and embedding crates.
- **[Testing](docs/TESTING.md)**, **[Reporting Standards](docs/REPORTING.md)**, & **[Upstream Pin Policy](docs/UPSTREAM.md)** — parity rules, numerical budgets, and weekly upstream monitoring.
- **[Changelog](CHANGELOG.md)** — full release history (`v0.1.0` → `v0.7.0`) and links to per-version release notes.

---

## Roadmap

1. **OpenWebText (OWT) Qualification (Upstream-Gated):** config parsing and synthetic operators are already verified; full inference, serving, and LoRA will be qualified once `HazyResearch/ncm` publishes OWT weights (monitored by the weekly upstream watcher).
2. **Serving v3 & NCM-Native Progressive Streaming:** live coarse-to-fine level streaming (`ncm run --progressive` and optional SSE level-refinement events), grouped multi-adapter micro-batching over a shared base forward pass, dynamic adapter load/unload API endpoints, and `/metrics`.
3. **Training v4 (True QLoRA & 860M Fine-Tuning on 8 GiB GPUs):** quantized frozen-base QLoRA training + activation gradient checkpointing (enabling `lm1b-medium` and `lm1b-large` LoRA fine-tuning inside 8 GiB VRAM), frozen `post_upsample_convs` in the LM1B LoRA forward pass, and `ncm merge-adapter`.
4. **Built-in `ncm bench` & Kernel Generalization:** one-command cross-precision latency/VRAM/throughput benchmarking CLI and wider FlashAttention head dimensions for future upstream model configs.

---

## Build, Local Checks & Workspace Layout

Requires Rust **1.99.1+** (edition 2024), `rustfmt`, `clippy`, and Python 3 for local validation (default CPU checks need no GPU SDK):

```sh
./tools/check.sh
# Linux/CUDA (real NVIDIA GPU required):
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
# Apple Silicon/Metal (run ./tools/setup-mlx-native.sh once first):
./tools/check.sh --mlx
```

| Path | Responsibility |
| --- | --- |
| `crates/ncm-core`, `crates/ncm-model` | Validated configs, `ComputePrecision`, backend traits, shared codec, generators & `RuntimeLoraAdapter` |
| `crates/ncm-io`, `crates/ncm-tokenizers` | Pure-Rust PyTorch/YAML/safetensors I/O, parallel SHA-256 verification, prequant sidecars, `ncm-lora-adapter-v1`, canonical & offline tokenizers |
| `crates/ncm-serve` | Zero-new-dep HTTP/1.1 + SSE/NDJSON OpenAI & Ollama server (`/api/ps`, dynamic micro-batching, adapter registry) |
| `crates/ncm-train` | Native Rust codec/generator training & multi-profile LoRA (`--save-adapter`, `BF16`, gradient accumulation, JSONL) on CPU, CUDA, and Metal |
| `crates/ncm-backend-cuda`, `crates/ncm-backend-mlx` | Native CUDA/cuDNN (FlashAttention-2, GGML `Q8_0`/`Q4_0`) and Apple Silicon MLX (Fast SDPA, group-64 affine) adapters |
| `crates/ncm-cli` | Single binary `ncm` (`doctor`, `pull`, `quantize`, `run`, `serve`, `train`, `sample`, `list`, `inspect`, `generate`, `reconstruct`) |
| `tools/` | `check.sh`, `setup-mlx-native.sh`, `package.sh`, `install.sh`, `render_homebrew_formula.py`, offline reference & benchmark suites |

---

## Provenance, Citation & License

- **License:** MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE). No model weights, datasets, or prebuilt binaries are committed to Git.
- **Qualified Upstream Pins ([upstream.toml](upstream.toml)):** `HazyResearch/ncm` @ `094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8`; `hazyresearch/ncm-checkpoints` @ `e34c3fab451ddbe7cf474cc97c44ab23321f2219` (legacy `c7027427` / `d342ce8a` artifacts remain backward-compatibly supported).
- **Upstream Paper Citation:**

```bibtex
@misc{kumbong2026nextconcept,
      title={Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
      author={Hermann Kumbong and Ishan S. Khare and Iddah Mlauzi and Andrew Woen and Charles Ding and E. Kelly Buchanan and Simran Arora and Azalia Mirhoseini and Christopher Ré},
      year={2026},
      eprint={2610.14162},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.14162},
}
```
