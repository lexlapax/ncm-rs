# ncm-rs

A Rust implementation of [HazyResearch/NCM](https://github.com/HazyResearch/ncm) — Next-Concept Modeling for parallel, coarse-to-fine generation — designed as a Python-free, verified foundation for running and serving local NCM models on **Linux/CUDA** and **Apple Silicon/MLX**. Production inference and HTTP serving use native CUDA/cuDNN or native MLX; Python is used strictly offline for checkpoint conversion, references, and evaluation.

**Status: `v0.4.1` released; Phase 8 & Phase 9 complete on Linux/CUDA and Apple Silicon/MLX.**
- **Phase 8 & Phase 9 complete on `main`** ([Phase 8 & Phase 9 dual-platform report](docs/results/2026-11-23-phase8-phase9/README.md)):
  - **Native Low-Precision Kernels & Fixed-Matrix Resampling (8A):** eliminated per-layer FP32 casts in `bf16`/`fp16`/`q8`/`q4` on both CUDA and Metal (**1.33×–1.77×** faster on RTX 3060 Ti; **1.22×–1.72×** faster on M1 Ultra). Eager Rust CUDA `bf16` b=1 on `tinystories-15k-ema` (**27.45 ms, 253.6 MiB**) beats upstream Python `094f25ca` **with `torch.compile` + CUDA graphs** (**31.28 ms, 647.9 MiB**) by **1.14×** at **2.55× less VRAM**.
  - **Verified Pre-Quantized `Q8`/`Q4` Sidecars (`ncm quantize`, 8B):** offline `ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx` writes hash-verified sidecars (`3.06×–5.45×` smaller) that load **`4.79×–9.23×` faster on CUDA** (`lm1b-large` `q4` `3,861 → 418.5 ms`) and **`2.50×–4.75×` faster on Metal** (`2,436 → 512.5 ms`) with **100% byte-identical** outputs and fail-closed tamper detection.
  - **`ncm-train` v2: LoRA, Upstream TinyStories `--init-from`, `BF16`, Gradient Accumulation, JSONL & Validation Eval (9A/9B):** fine-tune 123M `tinystories-15k-ema-v1` natively in Rust with `--lora-rank 8` (`0.96%` trainable, merged cleanly into verified `local-ncm-v1` on export, **4-decimal loss parity** across RTX 3060 Ti & M1 Ultra), `--train-precision bf16` (`1.05×` faster & `~576 MiB` less peak VRAM on CUDA), `--grad-accum N`, `.jsonl` datasets with sliding-window chunking, and `--val-data` / `--eval-every`.
- **`v0.4.1` published** (source-only, [upstream `094f25ca` / `e34c3fab` qualification](docs/results/2026-11-23-upstream-qualification/README.md)): dual-platform qualified TinyStories 15k EMA release (`0/2,564` code diffs, `0/512` ID diffs on CUDA & Metal), trimmed inference config support, explicit dual pin-pair verification (legacy `c7027427`/`d342ce8a` preserved), and updated `ncm pull tinystories` / `upstream.toml`.
- **`v0.4.0` published** (source-only, [Phase 7 report](docs/results/2026-11-23-phase7/README.md)):
  - **Native Weight Quantization (`--precision q8|q4`):** native `GGML Q8_0`/`Q4_0` on CUDA (`QTensor`/`QMatMul`) and group-64 affine `8-bit`/`4-bit` on Metal (`mlx_rs::ops::quantize`/`quantized_matmul`). Cuts `lm1b-large` b=1 VRAM **-68.1% (`q8`, 1,025 MiB, 1.24× faster)** / **-79.1% (`q4`, 673 MiB, 1.27× faster)** on RTX 3060 Ti (**b=16 in 975 MiB**!) and **-66.9% / -77.2% MLX peak** (-90.2% RSS) on M1 Ultra; `sudoku-easy` **128/128 `q8`** / **126–127/128 `q4`**; TinyStories fluent in both `q8` & `q4`.
  - **Ollama-Class CLI & Serving v2:** `ncm run MODEL [PROMPT]` (one-shot or hot-GPU interactive REPL), `ncm pull (--list | MODEL)`, `GET /api/ps` resident model inspection, and dynamic micro-batching (`ncm serve --max-batch 8`) delivering **2.21× CUDA / 1.85× Metal** concurrent throughput on `wmt14` (`3,560.8 tok/s` CUDA) and **1.64× CUDA / 1.97× Metal** on `lm1b-large` Q8.
  - **Native Rust Training & Fine-Tuning (`crates/ncm-train` + `ncm train`):** zero new external crates; pure-Rust CPU, CUDA (Candle), and Metal (MLX `value_and_grad`) reverse-mode autodiff for RVQ/STE/EMA codec training (`--stage codec`), frozen-codec generator training & fine-tuning (`--stage generator` / `--init-from` / `--stage all`), deterministic checkpoint save/resume, and direct verified `local-ncm-v1` export (**100% 8/8 sequence recovery** through `ncm inspect`, `ncm sample`, and `ncm serve` on both CUDA and Metal).
- **`v0.3.0` published:** opt-in `BF16`/`FP16` half precision, zero-new-dep OpenAI + Ollama HTTP/1.1 server (`ncm serve`, `ncm list`), inline `--prompt` + `--plain`, and [5-Minute Quickstart (`docs/QUICKSTART.md`)](docs/QUICKSTART.md).
- **`v0.2.0` published:** streaming verified loading + pipelined SHA-256, FP32 prefix K/V caching, self-contained `lm1b-*-inference-v2`, and config-driven `local-ncm-v1`.
- See the [Phase 8 & Phase 9 report](docs/results/2026-11-23-phase8-phase9/README.md), [Phase 7 report](docs/results/2026-11-23-phase7/README.md), [Phase 6 precision & serving report](docs/results/2026-11-22-phase6-precision-serving/README.md), [v0.2.0 optimization report](docs/results/2026-11-20-phase5-optimization/README.md), and [current state](agent-context-map.md).

## Quick Navigation

- **New users (5-minute build → `ncm pull` → `ncm run` → OpenAI/Ollama `ncm serve` → `ncm train`):** [5-Minute Quickstart (`docs/QUICKSTART.md`)](docs/QUICKSTART.md).
- **Full CLI & reproducible exact-hash setup:** [Setup & Usage Guide (`docs/USAGE.md`)](docs/USAGE.md) — verified from fresh clones on [Linux](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md) and [macOS](docs/results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md).
- **Developers (Rust library embedding, backend traits, `ncm-serve`, `ncm-train`, custom local models):** [Developer Guide (`docs/DEVELOPING.md`)](docs/DEVELOPING.md) and [Architecture (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md).
- **Release notes:** [v0.4.1 (published)](docs/RELEASE-NOTES-v0.4.1.md) | [v0.4.0 (published)](docs/RELEASE-NOTES-v0.4.0.md) | [v0.3.0 (published)](docs/RELEASE-NOTES-v0.3.0.md) | [v0.2.0 (published)](docs/RELEASE-NOTES-v0.2.0.md) | [v0.1.0 (published)](docs/RELEASE-NOTES-v0.1.0.md).

## Profile and evidence matrix

Results below apply to the tested hardware (RTX 3060 Ti on Linux, M1 Ultra on macOS), settings, and workloads; linked reports disclose measurement limits and earlier failures.

| Profile | Qualified scope | Quality and performance evidence |
| --- | --- | --- |
| Sudoku easy, medium, hard | FP32 greedy + opt-in BF16/FP16, CUDA and Metal | [6,000-board quality and timing](docs/results/2026-11-14-phase3-sudoku/README.md); [Phase 6 half precision](docs/results/2026-11-22-phase6-precision-serving/README.md) (128/128 exact in BF16/FP16 on both GPUs). |
| TinyStories | Seeded sampling (`FP32`/`BF16`/`Q8`/`Q4`; 15k EMA `e34c3fab` + legacy `d342ce8a`), CUDA and Metal; scoped exceptions | [Upstream `094f25ca` qualification](docs/results/2026-11-23-upstream-qualification/README.md) (`0/2,564` code diffs, `0/512` ID diffs on both GPUs); [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition) (FP16 explicitly rejected at load due to `4.37e10` / `2.48e10` codebook outliers). |
| XSum | Top-1, CFG 0 + opt-in BF16/FP16 + serving, CUDA and Metal | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition), [v0.2.0 prefix cache](docs/results/2026-11-20-phase5-optimization/README.md), and [Phase 6 precision & serving](docs/results/2026-11-22-phase6-precision-serving/README.md). |
| WMT14 | Scoped top-1 (CFG 0 and 1) + opt-in BF16/FP16 + serving, CUDA and Metal | [CLI qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification), [v0.2.0 prefix cache](docs/results/2026-11-20-phase5-optimization/README.md), and [Phase 6 precision & serving](docs/results/2026-11-22-phase6-precision-serving/README.md). |
| LM1B small, medium, large | Seeded generation (`v1` + self-contained `v2`) + opt-in BF16/FP16 + serving, CUDA and Metal | [Quality campaign](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md), [v0.2.0 vs Python MLX](docs/results/2026-11-20-phase5-optimization/README.md), and [Phase 6 precision & serving](docs/results/2026-11-22-phase6-precision-serving/README.md) (`lm1b-large` b=16 in 1.96 GiB CUDA VRAM). |
| Custom local NCM (`local-ncm-v1`) | Config-driven greedy & seeded + serving, CPU, CUDA, and Metal | End-to-end verified on CPU, CUDA, and Metal; see [docs/DEVELOPING.md](docs/DEVELOPING.md#4-running-your-own-local-ncm-model-local-ncm-v1) and [docs/ARTIFACTS.md](docs/ARTIFACTS.md). |
| OpenWebText | Config and synthetic operators only | [Coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md); weights are unreleased and inference is refused. |

## CLI, local serving, and native training

Run `ncm <command> --help` (`sample`, `run`, `pull`, `quantize`, `serve`, `train`, `list`) for full syntax and explicitly gated experimental options.

```sh
ncm list [--models-dir DIR] [--json]
ncm pull --list | ncm pull wmt14 [--dry-run]
ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx [--force]
ncm run wmt14 ["Guten Morgen!"] [--precision fp32|bf16|fp16|q8|q4]
ncm serve [--models-dir DIR] [--precision fp32|bf16|fp16|q8|q4] [--port 11434] [--max-batch 8] [--preload MODEL]
ncm train (--config cfg.json | --init-from ARTIFACT) --data train.jsonl --output out --device cuda|mlx|cpu \
  [--lora-rank 8 --lora-alpha 16] [--train-precision fp32|bf16] [--grad-accum 2] [--val-data val.jsonl --eval-every 10]
ncm inspect ARTIFACT
ncm sample ARTIFACT DEVICE --prompt "Hello world" [--precision fp32|bf16|fp16|q8|q4] [--plain]
ncm generate|reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]  # Sudoku
```

- **Default FP32 byte-identity:** `--precision fp32` (default) preserves exact parity across all 9 profiles on both CUDA and Metal.
- **Opt-in half precision & quantization:** `--precision fp16` (XSum/WMT14/LM1B/Sudoku), `--precision bf16` (TinyStories / wide-range), and `--precision q8|q4` (native `GGML Q8_0`/`Q4_0` on CUDA, group-64 affine on Metal) work directly on existing FP32 artifacts.
- **Local HTTP serving (`ncm serve`):** localhost-only by default (`127.0.0.1:11434`), serving both OpenAI (`/v1/models`, `/v1/completions`, `/v1/chat/completions`) and Ollama (`/`, `/health`, `/api/tags`, `/api/ps`, `/api/show`, `/api/generate`, `/api/chat`) clients with SSE/NDJSON streaming, dynamic micro-batching (`--max-batch`), and a bounded LRU GPU cache (`--max-loaded`).
- **RNG & experimental flags:** neither runtime reproduces PyTorch's RNG stream. Experimental flags (`--experimental-stochastic-xsum`, `--experimental-positive-cfg-xsum`, `--experimental-cfg-wmt14`, LM1B Metal `--top-p-fp32`, WMT14 `mlx-retry-padded`) require explicit opt-in. OWT inference is refused.
- **Provenance note:** no model weights, evaluation datasets, or prebuilt binaries are committed to Git.

## Build and local checks

Rust **1.99.1+** (edition 2024), `rustfmt`, `clippy`, and Python 3 are required for local checks. Default CPU checks require no GPU SDK. For quickstart or full CUDA/cuDNN and Apple Silicon MLX setup steps, follow [docs/QUICKSTART.md](docs/QUICKSTART.md) or [docs/USAGE.md](docs/USAGE.md).

```sh
./tools/check.sh
# Linux (tested RTX 3060 Ti configuration; real GPU required):
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
# Apple Silicon (requires MLX_SYS_PREBUILT and rpath per docs/USAGE.md):
./tools/check.sh --mlx
```

Checks never download weights or launch SSH jobs. Build CI is intentionally disabled; the metadata-only weekly upstream watcher remains enabled.

## Workspace and project documents

| Path | Responsibility |
| --- | --- |
| `crates/ncm-core`, `crates/ncm-model` | Validated configs, `ComputePrecision`, backend traits, shared codec & generators |
| `crates/ncm-tokenizers`, `crates/ncm-io` | Pinned & offline `local-ncm-v1` tokenizers, parallel SHA-256 & streaming verified I/O |
| `crates/ncm-serve` | Zero-new-dep HTTP/1.1 + SSE/NDJSON OpenAI & Ollama server (`/api/ps`, dynamic micro-batching) and model registry |
| `crates/ncm-train` | Native Rust training & fine-tuning (RVQ/EMA codec + frozen-codec generator) on CPU, CUDA (Candle), and Metal (MLX) |
| `crates/ncm-backend-cuda`, `crates/ncm-backend-mlx` | Explicit CUDA/cuDNN and native MLX/Metal runtime adapters (`fp32`, `bf16`, `fp16`, `q8`, `q4`) |
| `crates/ncm-cli` | Unified `ncm run` / `pull` / `serve` / `train` / `sample` / `list` / `inspect` / `generate` / `reconstruct` CLI |
| `tools/reference`, `tests/fixtures`, `tools/bench` | Offline conversion, quickstart wrapper, pinned oracles, equivalence & benchmark suites |

Docs: [5-minute quickstart](docs/QUICKSTART.md), [usage guide](docs/USAGE.md), [developer guide](docs/DEVELOPING.md), [architecture](docs/ARCHITECTURE.md), [artifact contract](docs/ARTIFACTS.md), [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md), [results index](docs/results/README.md), and [plan](docs/PLAN.md). Agents start with [AGENTS.md](AGENTS.md) and [agent-context-map.md](agent-context-map.md).

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Qualified upstream code: `094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8`; checkpoint revision: `e34c3fab451ddbe7cf474cc97c44ab23321f2219` (legacy `c7027427` / `d342ce8a` artifacts remain supported). Model and dataset licenses are tracked separately.
