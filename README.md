# ncm-rs

A Rust implementation of [HazyResearch/NCM](https://github.com/HazyResearch/ncm) — Next-Concept Modeling for parallel, coarse-to-fine generation — designed as a Python-free, verified FP32 foundation for running local NCM models on **Linux/CUDA** and **Apple Silicon/MLX**. Production inference uses native CUDA/cuDNN or native MLX; Python is used strictly offline for checkpoint conversion, references, and evaluation.

**Status: `v0.2.0` released on Linux/CUDA and Apple Silicon/MLX.**
- **`v0.2.0` published** (source-only):
  - **Streaming verified weight loading + pipelined SHA-256:** **2.05×–2.46× faster load** and up to **-2.8 GiB (0.22×) host HWM** on CUDA; **1.93×–3.63× faster load** and **48–49% lower peak RSS** vs Python MLX on Metal (`lm1b-large`: 3,298 vs 6,495 MiB).
  - **FP32 prefix K/V caching (zero corpus diffs):** **3.24× faster XSum** (`155.8 → 48.1 ms`) and **1.54× faster WMT14** (`58.5 → 38.0 ms`) on CUDA; **2.42× faster XSum** (`204.6 → 84.6 ms`, `-187 MiB` MLX peak) and **1.20× faster WMT14** (`-59 MiB` MLX peak) on Metal.
  - **Local NCM model foundation & parity fixes:** config-driven `local-ncm-v1` artifacts + stdlib exporter (`tools/reference/export_local_ncm.py`), self-contained `lm1b-*-inference-v2` option, XSum Metal seeded + positive CFG support, and **9/9 byte-identical CLI outputs** on both GPUs.
- See the [Phase 5 final audit](docs/results/2026-11-20-phase5-audit/phase5-final-audit-v1.md), [v0.2.0 dual-platform optimization report](docs/results/2026-11-20-phase5-optimization/README.md), [cross-phase gap audit](docs/results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md), and [current state](agent-context-map.md).

## Quick Navigation

- **Users (install, build, fetch/export checkpoints, run inference):** [Setup & Usage Guide (`docs/USAGE.md`)](docs/USAGE.md) — verified from fresh clones on [Linux](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md) and [macOS](docs/results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md).
- **Developers (Rust library embedding, backend traits, adding local models):** [Developer Guide (`docs/DEVELOPING.md`)](docs/DEVELOPING.md) and [Architecture (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md).
- **Release notes:** [v0.2.0 (published)](docs/RELEASE-NOTES-v0.2.0.md) | [v0.1.0 (published)](docs/RELEASE-NOTES-v0.1.0.md).

## Profile and evidence matrix

Results below apply to the tested hardware (RTX 3060 Ti on Linux, M1 Ultra on macOS), settings, and workloads; linked reports disclose measurement limits and earlier failures.

| Profile | Qualified scope | Quality and performance evidence |
| --- | --- | --- |
| Sudoku easy, medium, hard | FP32 greedy, CUDA and Metal | [6,000-board quality and timing](docs/results/2026-11-14-phase3-sudoku/README.md); CUDA has one intermediate near-tie code difference, with no final-ID difference. |
| TinyStories | Seeded sampling, CUDA and Metal; scoped exceptions | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition); [accepted Metal numerical and latency-tail exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md). |
| XSum | Top-1, CFG 0, CUDA and Metal; scoped exceptions | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition); [Metal output differences](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md) and [CUDA stochastic exceptions](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md). |
| WMT14 | Scoped top-1 (CFG 0 and 1), CUDA and Metal | [CLI and profile qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification), including quality, timing, memory and retained Metal intermediate differences. |
| LM1B small, medium, large | Seeded generation, CUDA and Metal; approved 80-seed gate (`v1` + self-contained `v2`) | [Quality campaign](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md), [CUDA timing](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md), [Metal timing](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md#latency-and-memory-against-python-mlx), [v0.2.0 load, warm & RSS vs Python MLX](docs/results/2026-11-20-phase5-optimization/README.md), and [scoped user decisions](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md). |
| Custom local NCM (`local-ncm-v1`) | Config-driven FP32 greedy & seeded, CUDA and Metal | End-to-end verified on CPU, CUDA, and Metal; see [docs/DEVELOPING.md](docs/DEVELOPING.md#4-running-your-own-local-ncm-model-local-ncm-v1) and [docs/ARTIFACTS.md](docs/ARTIFACTS.md). |
| OpenWebText | Config and synthetic operators only | [Coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md); weights are unreleased and inference is refused. |

## CLI and current limitations

The explicit-device CLI accepts JSON arrays of token-ID rows or `--text` string arrays. `ncm sample` dispatches on the artifact manifest's profile; legacy `sample-*` commands remain as aliases. Run `ncm --help` or `ncm sample --help` for full syntax and explicitly gated experimental options.

```sh
ncm inspect ARTIFACT
ncm sample ARTIFACT DEVICE INPUT_JSON [--seed N] [--cfg S] [--decode]
ncm sample ARTIFACT DEVICE TEXT_JSON --text [--decode]  # LM1B accepts empty strings only
ncm generate ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]     # Sudoku
ncm reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]  # Sudoku codec
```

- **Byte-identical across all 9 profiles** (`baseline == alias == sample`) on both CUDA and Metal (`v0.2.0` candidate `e7b8cfa`).
- **Text input & decode:** `--decode` works across all text profiles and `local-ncm-v1` (when `tokenizer.json` is bundled). TinyStories `--text` requires strings encoding to exactly one non-padding token; XSum and WMT14 truncate/pad to 512 and 96 tokens respectively; LM1B accepts `[101]` ID rows or 1–32 empty strings under `--text`. LM1B `v1` requires the sibling rotary sidecar (`lm1b-reference-rope-v1.safetensors`) or `--reference-rope FILE`; LM1B `v2` bundles the verified rotary tables inside `inference.safetensors`.
- **Precision & scope:** FP32 only in v0.1/v0.2 (BF16/FP16 and quantization are deferred). OWT inference is refused. No serving API, streaming, cancellation, or concurrent-request throughput is qualified.
- **RNG & experimental flags:** neither runtime reproduces PyTorch's RNG stream (Metal/CUDA draw-level differences are a [scoped LM1B exception](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md#2-metal-stochastic-draw-differences-scoped-exception)). XSum stochastic sampling and positive CFG, WMT14 CFG scales other than 0/1 (`--experimental-cfg-wmt14`), LM1B Metal `--top-p-fp32`, and WMT14 `mlx-retry-padded` require explicit opt-in flags/device names.
- **Memory & batch limits:** on the tested 8 GiB CUDA card, pinned Python LM1B-large OOMed at batch 16 (Rust was timed at batches 1/4/8 alongside Python). CLI argument limits do not guarantee that a batch fits in available VRAM.
- **Provenance note:** the historical Sudoku medium/hard exporter script is unresolved; provenance-only re-exports produce identical weights and 2,000-board outputs ([export investigation](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md)). No model weights, evaluation datasets, or prebuilt binaries are committed to Git.

## Build and local checks

Rust **1.99.1+** (edition 2024), `rustfmt`, `clippy`, and Python 3 are required for local checks. Default CPU checks require no GPU SDK. For full CUDA/cuDNN and Apple Silicon MLX setup steps, follow [docs/USAGE.md](docs/USAGE.md).

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
| `crates/ncm-core`, `crates/ncm-model` | Validated configs, backend traits, shared FP32 codec & generators |
| `crates/ncm-tokenizers`, `crates/ncm-io` | Pinned tokenizers, parallel SHA-256 & streaming verified artifact I/O |
| `crates/ncm-backend-cuda`, `crates/ncm-backend-mlx` | Explicit CUDA/cuDNN and native MLX/Metal runtime adapters |
| `crates/ncm-cli` | Unified `ncm sample` / `inspect` / `generate` / `reconstruct` CLI |
| `tools/reference`, `tests/fixtures`, `tools/bench` | Offline conversion, pinned oracles, equivalence & benchmark suites |

Docs: [usage guide](docs/USAGE.md), [developer guide](docs/DEVELOPING.md), [architecture](docs/ARCHITECTURE.md), [artifact contract](docs/ARTIFACTS.md), [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md), [results index](docs/results/README.md), and [plan](docs/PLAN.md). Agents start with [AGENTS.md](AGENTS.md) and [agent-context-map.md](agent-context-map.md).

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Pinned upstream code: `c70274273f489abc73832b2b98584a8483b6b203`; checkpoint revision: `d342ce8aaea58830eef3b8ad473664673e41f692`. Model and dataset licenses are tracked separately.
