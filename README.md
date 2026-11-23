# ncm-rs

A Rust implementation of [HazyResearch/NCM](https://github.com/HazyResearch/ncm) — Next-Concept Modeling for parallel, coarse-to-fine generation — designed as a Python-free, verified foundation for running and serving local NCM models on **Linux/CUDA** and **Apple Silicon/MLX**. Production inference and HTTP serving use native CUDA/cuDNN or native MLX; Python is used strictly offline for checkpoint conversion, references, and evaluation.

**Status: `v0.3.0` released on Linux/CUDA and Apple Silicon/MLX.**
- **`v0.3.0` published** (source-only):
  - **Opt-in `BF16` / `FP16` half precision (`--precision fp32|bf16|fp16`):** cuts CUDA VRAM **40.7%–48.5%** and MLX peak **36.4%–47.3%** (host RSS up to **-90.2%**), runs up to **1.47× faster** on CUDA (`lm1b-large`: `246.2 → 167.7 ms`) and **1.15× faster** on Metal (`lm1b-large` b=16), and fits **`lm1b-large` batch 16 in 1,955.1 MiB VRAM** on an 8 GiB RTX 3060 Ti — while keeping default `--precision fp32` 100% untouched.
  - **Drop-in OpenAI & Ollama local HTTP server (`ncm serve`, `crates/ncm-serve`):** zero new external crate dependencies; exposes OpenAI (`/v1/models`, `/v1/completions`, `/v1/chat/completions` + SSE streaming) and Ollama (`/`, `/health`, `/api/tags`, `/api/show`, `/api/generate`, `/api/chat` + NDJSON streaming) on `127.0.0.1:11434` with auto-discovered models and an LRU GPU worker cache.
  - **Newbie ergonomics:** [5-Minute Quickstart (`docs/QUICKSTART.md`)](docs/QUICKSTART.md), one-command `tools/reference/quickstart_export.py`, `ncm list`, and inline `--prompt "..."` on `ncm sample`.
- **`v0.2.0` published baseline:** streaming verified loading + pipelined SHA-256 (**2.05×–2.46×** CUDA / **1.93×–3.63×** Metal faster load, **48–49% lower peak RSS** than Python MLX), FP32 prefix K/V caching (**3.24×** CUDA / **2.42×** Metal faster XSum, **0** corpus diffs), self-contained `lm1b-*-inference-v2`, and config-driven `local-ncm-v1`.
- See the [Phase 6 precision & serving report](docs/results/2026-11-22-phase6-precision-serving/README.md), [v0.2.0 optimization report](docs/results/2026-11-20-phase5-optimization/README.md), [Phase 5 final audit](docs/results/2026-11-20-phase5-audit/phase5-final-audit-v1.md), and [current state](agent-context-map.md).

## Quick Navigation

- **New users (5-minute build → export → `--prompt` → OpenAI/Ollama `ncm serve`):** [5-Minute Quickstart (`docs/QUICKSTART.md`)](docs/QUICKSTART.md).
- **Full CLI & reproducible exact-hash setup:** [Setup & Usage Guide (`docs/USAGE.md`)](docs/USAGE.md) — verified from fresh clones on [Linux](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md) and [macOS](docs/results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md).
- **Developers (Rust library embedding, backend traits, `ncm-serve`, adding local models):** [Developer Guide (`docs/DEVELOPING.md`)](docs/DEVELOPING.md) and [Architecture (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md).
- **Release notes:** [v0.3.0 (published)](docs/RELEASE-NOTES-v0.3.0.md) | [v0.2.0 (published)](docs/RELEASE-NOTES-v0.2.0.md) | [v0.1.0 (published)](docs/RELEASE-NOTES-v0.1.0.md).

## Profile and evidence matrix

Results below apply to the tested hardware (RTX 3060 Ti on Linux, M1 Ultra on macOS), settings, and workloads; linked reports disclose measurement limits and earlier failures.

| Profile | Qualified scope | Quality and performance evidence |
| --- | --- | --- |
| Sudoku easy, medium, hard | FP32 greedy + opt-in BF16/FP16, CUDA and Metal | [6,000-board quality and timing](docs/results/2026-11-14-phase3-sudoku/README.md); [Phase 6 half precision](docs/results/2026-11-22-phase6-precision-serving/README.md) (128/128 exact in BF16/FP16 on both GPUs). |
| TinyStories | Seeded sampling (FP32 + BF16), CUDA and Metal; scoped exceptions | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition); [Phase 6 precision](docs/results/2026-11-22-phase6-precision-serving/README.md) (FP16 explicitly rejected at load due to `2.48e10` codebook entry). |
| XSum | Top-1, CFG 0 + opt-in BF16/FP16 + serving, CUDA and Metal | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition), [v0.2.0 prefix cache](docs/results/2026-11-20-phase5-optimization/README.md), and [Phase 6 precision & serving](docs/results/2026-11-22-phase6-precision-serving/README.md). |
| WMT14 | Scoped top-1 (CFG 0 and 1) + opt-in BF16/FP16 + serving, CUDA and Metal | [CLI qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification), [v0.2.0 prefix cache](docs/results/2026-11-20-phase5-optimization/README.md), and [Phase 6 precision & serving](docs/results/2026-11-22-phase6-precision-serving/README.md). |
| LM1B small, medium, large | Seeded generation (`v1` + self-contained `v2`) + opt-in BF16/FP16 + serving, CUDA and Metal | [Quality campaign](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md), [v0.2.0 vs Python MLX](docs/results/2026-11-20-phase5-optimization/README.md), and [Phase 6 precision & serving](docs/results/2026-11-22-phase6-precision-serving/README.md) (`lm1b-large` b=16 in 1.96 GiB CUDA VRAM). |
| Custom local NCM (`local-ncm-v1`) | Config-driven greedy & seeded + serving, CPU, CUDA, and Metal | End-to-end verified on CPU, CUDA, and Metal; see [docs/DEVELOPING.md](docs/DEVELOPING.md#4-running-your-own-local-ncm-model-local-ncm-v1) and [docs/ARTIFACTS.md](docs/ARTIFACTS.md). |
| OpenWebText | Config and synthetic operators only | [Coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md); weights are unreleased and inference is refused. |

## CLI and local serving

Run `ncm --help`, `ncm sample --help`, `ncm serve --help`, or `ncm list --help` for full syntax and explicitly gated experimental options.

```sh
ncm list [--models-dir DIR] [--json]
ncm inspect ARTIFACT
ncm sample ARTIFACT DEVICE --prompt "Hello world" [--precision fp32|bf16|fp16]
ncm sample ARTIFACT DEVICE INPUT_JSON [--seed N] [--cfg S] [--precision fp32|bf16|fp16] [--decode]
ncm sample ARTIFACT DEVICE TEXT_JSON --text [--decode]  # LM1B accepts empty strings only
ncm serve [--models-dir DIR] [--device DEV] [--precision fp32|bf16|fp16] [--port 11434] [--preload MODEL]
ncm generate|reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]  # Sudoku
```

- **Default FP32 byte-identity:** `--precision fp32` (default) preserves exact parity across all 9 profiles on both CUDA and Metal.
- **Opt-in half precision:** `--precision fp16` (recommended for XSum, WMT14, LM1B, Sudoku) and `--precision bf16` (recommended for TinyStories or wide-range weights) work directly on existing FP32 artifacts; out-of-range FP16 weights (`> 65504.0`) fail explicitly on load.
- **Local HTTP serving (`ncm serve`):** localhost-only by default (`127.0.0.1:11434`), serving both OpenAI (`/v1/models`, `/v1/completions`, `/v1/chat/completions`) and Ollama (`/`, `/health`, `/api/tags`, `/api/show`, `/api/generate`, `/api/chat`) clients with SSE/NDJSON streaming, bounded 4 MiB request bodies, and a serialized GPU worker thread with bounded LRU cache (`--max-loaded`).
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
| `crates/ncm-serve` | Zero-new-dep HTTP/1.1 + SSE/NDJSON OpenAI & Ollama server and model registry |
| `crates/ncm-backend-cuda`, `crates/ncm-backend-mlx` | Explicit CUDA/cuDNN and native MLX/Metal runtime adapters (`fp32`, `bf16`, `fp16`) |
| `crates/ncm-cli` | Unified `ncm sample` / `serve` / `list` / `inspect` / `generate` / `reconstruct` CLI |
| `tools/reference`, `tests/fixtures`, `tools/bench` | Offline conversion, quickstart wrapper, pinned oracles, equivalence & benchmark suites |

Docs: [5-minute quickstart](docs/QUICKSTART.md), [usage guide](docs/USAGE.md), [developer guide](docs/DEVELOPING.md), [architecture](docs/ARCHITECTURE.md), [artifact contract](docs/ARTIFACTS.md), [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md), [results index](docs/results/README.md), and [plan](docs/PLAN.md). Agents start with [AGENTS.md](AGENTS.md) and [agent-context-map.md](agent-context-map.md).

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Pinned upstream code: `c70274273f489abc73832b2b98584a8483b6b203`; checkpoint revision: `d342ce8aaea58830eef3b8ad473664673e41f692`. Model and dataset licenses are tracked separately.
