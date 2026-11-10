# ncm-rs

A planned Rust port of [HazyResearch/NCM](https://github.com/HazyResearch/ncm):
Next-Concept Modeling for parallel, coarse-to-fine text generation.

**Status: Phase 1 approved and in progress.** Linux/CUDA candidate operators pass
18 synthetic PyTorch parity cases. A scoped Sudoku checkpoint exporter, checked
Rust artifact loader, and Sudoku text tokenizer now exist; there is no complete
Rust model or inference command yet. Within each phase, Linux/CUDA is validated first, then
macOS/MLX. Both gates are required before advancing phases.

## Goals

- Shared, idiomatic Rust model logic and Python-free production inference.
- Apple Silicon: **MLX through `mlx-rs`**, not Candle's Metal backend.
- Linux/NVIDIA: **Candle CUDA**, with specialized kernels only where profiling warrants them.
- Checkpoint and numerical parity first; measured speed improvements second.
- Eventually port training, separately from inference compatibility.

“Rust as much as possible” is not “no native libraries”: MLX uses C/C++/Metal,
CUDA uses NVIDIA libraries/kernels. Python is confined to offline conversion,
reference fixtures, and initially external evaluation oracles.

## Read the plan

1. [Research and upstream audit](docs/RESEARCH.md)
2. [Architecture and runtime decisions](docs/ARCHITECTURE.md)
3. [Phased port plan and acceptance gates](docs/PLAN.md)
4. [Testing and performance protocol](docs/TESTING.md)
5. [Checkpoint/artifact contract proposal](docs/ARTIFACTS.md)
6. [Periodic upstream compatibility](docs/UPSTREAM.md)
7. [Quality/performance reporting and publication](docs/REPORTING.md)

Agents: start with [AGENTS.md](AGENTS.md) and its lazy-loading
[context map](agent-context-map.md); update the map before every commit.

The first vertical slice is **Sudoku on both runtimes**, followed by TinyStories,
XSum, WMT14, and LM1B. OpenWebText checkpoint testing is blocked: the inspected
Hugging Face revision contains only `owt/TODO.md`. Upstream has model-side training
pieces but no end-to-end training pipeline; training reproduction is a later project.

## Workspace

| Path | Responsibility / current scope |
| --- | --- |
| `crates/ncm-core` | Validated configuration, tensor/backend contract, errors |
| `crates/ncm-model` | Shared VQ codec, transformer, generation and sampling |
| `crates/ncm-tokenizers` | HF tokenizers, TinyStories byte BPE, Sudoku |
| `crates/ncm-io` | Versioned manifests, safetensors, artifact validation |
| `crates/ncm-backend-cuda` | Candidate Candle CPU/CUDA operators, fixture tests and microprobe |
| `crates/ncm-backend-mlx` | Apple Silicon MLX adapter |
| `crates/ncm-cli` | Future `ncm` command; currently a status-only executable |
| `tools/reference` | Pinned operator/model oracles and scoped Sudoku checkpoint exporter |
| `tests/fixtures` | 18 synthetic FP32 operator cases, Sudoku tokenizer goldens and fixture policy |
| `tools/bench` | Validated aggregation of raw probe results |
| `benches` | Benchmark protocol and reproduction links |

Candle is pinned to 0.11.1 behind the backend crate's opt-in `cpu`/`cuda` features.
Default builds do not require a GPU SDK. MLX integration and a shared backend
contract remain provisional; the status-only CLI has no runtime features yet.

[First public operator comparison](docs/results/2026-11-09-cuda-operators/README.md):
correctness passes, but composed resampling is substantially slower than PyTorch.
These tiny microprobes establish no NCM quality or generation speedup.

## Local checks

Rust **1.99.1+** (edition 2024), rustfmt, Clippy and Python 3. This is a
conservative tested toolchain floor, not a claim that every older version fails.
The earlier Rust 1.88 claim was invalid: Candle 0.11.1 uses x86 intrinsics that
are unstable on that compiler. Build CI has been replaced with local validation
at the user's request; only the metadata-only upstream watcher runs on GitHub.

```sh
./tools/check.sh
# Optional: additionally validate on the local NVIDIA GPU (tested Linux setup).
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  ./tools/check.sh --cuda
```

The script works from any working directory, fails on the first error, does not
install tools or download weights, and never launches SSH jobs. The default runs
CPU fixtures and report-tool tests; `--cuda` additionally runs the real-device
operator probe and writes a fresh report under `reports/`. It needs no Python
Torch installation. A smoke report is not a full three-run benchmark comparison.

The [Sudoku artifact report](docs/results/2026-11-10-sudoku-artifacts/README.md)
records byte-exact conversion, CPU/CUDA tensor readback and Python model traces.
It does not establish Rust model parity or load-performance qualification. `macstudio` is now available for the Apple pass; announce MLX work
when Linux Phase 1 is ready and recheck availability in later sessions.

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Research is pinned to upstream
`c70274273f489abc73832b2b98584a8483b6b203`. Model and dataset licenses must be tracked
separately; no weights or evaluation data are bundled.
