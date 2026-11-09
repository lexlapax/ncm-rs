# ncm-rs

A planned Rust port of [HazyResearch/NCM](https://github.com/HazyResearch/ncm):
Next-Concept Modeling for parallel, coarse-to-fine text generation.

**Status: research and workspace scaffold only. No model, GPU backend, checkpoint
converter, or inference command is implemented. Implementation awaits plan review.**

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

| Path | Intended responsibility (not implemented) |
| --- | --- |
| `crates/ncm-core` | Validated configuration, tensor/backend contract, errors |
| `crates/ncm-model` | Shared VQ codec, transformer, generation and sampling |
| `crates/ncm-tokenizers` | HF tokenizers, TinyStories byte BPE, Sudoku |
| `crates/ncm-io` | Versioned manifests, safetensors, artifact validation |
| `crates/ncm-backend-cuda` | Candle CPU reference and Linux CUDA adapter |
| `crates/ncm-backend-mlx` | Apple Silicon MLX adapter |
| `crates/ncm-cli` | Future `ncm` command; currently a status-only executable |
| `tools/reference` | Offline Python oracle/exporter boundary |
| `tests/fixtures` | Fixture policy; no generated fixtures yet |
| `benches` | Benchmark protocol; no benchmarks yet |

No external dependencies are selected or linked in this scaffold. Runtime choices
are provisional until Phase 1's capability spikes. There are no GPU feature flags yet.

## Scaffold checks

Rust 1.88+ (edition 2024); CI checks the declared minimum and current stable.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo run -p ncm-cli --locked
```

These commands validate the **scaffold only**; empty library crates have no model
tests. GPU verification will require real Apple Silicon and NVIDIA runners.

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Research is pinned to upstream
`c70274273f489abc73832b2b98584a8483b6b203`. Model and dataset licenses must be tracked
separately; no weights or evaluation data are bundled.
