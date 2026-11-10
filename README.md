# ncm-rs

A Rust port in progress of [HazyResearch/NCM](https://github.com/HazyResearch/ncm):
Next-Concept Modeling for parallel, coarse-to-fine text generation.

**Status: Phase 2 complete on Linux/CUDA and Apple Silicon/MLX, including performance remediation.**
The shared FP32 Sudoku codec and pinned TinyStories/GPT-2/Qwen/BERT tokenizers pass
[final correctness/performance/memory acceptance](docs/results/2026-11-10-phase2-acceptance/README.md):
642 trained traces / 2,500 decisions and 78 performance/memory comparisons on each
platform, with unchanged thresholds. CUDA codec latency and Apple loading/memory
deficits are remediated; original negative results remain published.
Text tokenizer support does not imply full text-model inference.
**No generator or inference CLI yet.** See the
[Phase 1 acceptance checklist](docs/PHASE1.md) and [Phase 2 contract](docs/PHASE2.md).
Within each phase, Linux/CUDA is validated
first, then macOS/MLX; both gates are required before advancing.

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
| `crates/ncm-core` | Validated codec configuration, provisional backend contract, errors |
| `crates/ncm-model` | Shared FP32 Sudoku codec and blocks; generation/sampling planned |
| `crates/ncm-tokenizers` | Sudoku, pinned HF tokenizers and TinyStories byte BPE |
| `crates/ncm-io` | Versioned manifests, safetensors, artifact validation |
| `crates/ncm-backend-cuda` | Candle CUDA codec/operators; CPU fixture path and real-device checks |
| `crates/ncm-backend-mlx` | Native MLX codec/operators, real-Metal tests and memory probes |
| `crates/ncm-cli` | Future `ncm` command; currently a status-only executable |
| `tools/reference` | Pinned operator/model oracles and scoped Sudoku checkpoint exporter |
| `tests/fixtures` | 18 synthetic FP32 operator cases, Sudoku tokenizer goldens and fixture policy |
| `tools/bench` | Validated aggregation of raw probe results |
| `benches` | Benchmark protocol and reproduction links |

Candle is pinned to 0.11.1 behind the backend crate's opt-in `cpu`/`cuda` features.
Default builds do not require a GPU SDK. MLX is opt-in on Apple Silicon with a
pinned native runtime. The shared backend contract is validated for the supported
codec; generator APIs remain future work. The status-only CLI has no runtime features yet.

[First public operator comparison](docs/results/2026-11-09-cuda-operators/README.md):
correctness passes, but composed resampling is substantially slower than PyTorch.
These tiny microprobes establish no NCM quality or generation speedup.
[Apple Phase 1 results](docs/results/2026-11-10-mlx-phase1/README.md) include the
negative finding that Rust copied loading is slower and has higher host RSS than
the adapted Python baseline.

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

On Apple Silicon, use `./tools/check.sh --mlx` with the
[documented native MLX environment](tools/reference/README.md#apple-silicon--mlx-phase-1).

The script works from any working directory, fails on the first error, does not
install tools or download weights, and never launches SSH jobs. The default runs
CPU fixtures and report-tool tests; `--cuda` additionally runs the real-device
operator probe and writes a fresh report under `reports/`. It needs no Python
Torch installation. A smoke report is not a full three-run benchmark comparison.

The [Sudoku artifact report](docs/results/2026-11-10-sudoku-artifacts/README.md)
records byte-exact conversion, CPU/CUDA tensor readback and Python model traces.
It does not establish Rust model parity. The later Linux and Apple reports close
Phase 1's capability/baseline gates. Phase 2 now adds qualified codec/tokenizer
parity and remediation; end-to-end generation quality remains future work. Recheck Mac availability before starting a later phase's hardware runs.

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Research is pinned to upstream
`c70274273f489abc73832b2b98584a8483b6b203`. Model and dataset licenses must be tracked
separately; no weights or evaluation data are bundled.
