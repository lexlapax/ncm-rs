# ncm-rs

A Rust port in progress of [HazyResearch/NCM](https://github.com/HazyResearch/ncm):
Next-Concept Modeling for parallel, coarse-to-fine text generation.

**Status: Phase 3 complete for greedy Sudoku on CUDA and MLX.**
The earlier Linux/CUDA and Apple Silicon/MLX remediation is accepted. The
[additional plan](docs/PHASE2-OPTIMIZATION-REVIEW.md#approved-five-step-phase-2-plan-and-decision-gates)
has since produced controlled **78/78** timing and host-memory audits on
[Linux](docs/results/2026-11-11-phase2-optimization/cuda-bounded-linux-gate-v1.md)
and [Apple](docs/results/2026-11-11-phase2-optimization/cuda-bounded-mlx-check-v1.md),
with quality checks passing and earlier failures retained. The new Linux candidate's
synchronized native CUDA-pool snapshots are below 1.05× prior Rust. The [final audit](docs/results/2026-11-11-phase2-optimization/phase2-final-audit-v1.md) records the accepted immediate B4 snapshot exception (1.058744× prior Rust), and separately discloses a 1.20127× B4 encode MLX active-memory peak; neither is host RSS.
The [Phase 3 comparison](docs/results/2026-11-14-phase3-sudoku/README.md) records
all three 2,000-board evaluations and generation latency, throughput and memory
on both platforms. Metal matched every original Python code and output; CUDA had
one intermediate near-tie code difference and no final output differences.
The earlier accepted FP32 Sudoku codec and pinned TinyStories/GPT-2/Qwen/BERT
tokenizers passed [correctness/performance/memory acceptance](docs/results/2026-11-10-phase2-acceptance/README.md):
642 trained traces / 2,500 decisions and 78 performance/memory comparisons on each
platform, with unchanged thresholds. CUDA codec latency and Apple loading/memory
deficits are remediated; original negative results remain published.
Text tokenizer support does not imply full text-model inference.
The greedy Sudoku generator and explicit-device CLI have passed the scoped
Phase 3 checks; a general inference release is not yet claimed. See the
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
| `crates/ncm-model` | Shared FP32 Sudoku codec, blocks and eager greedy generator |
| `crates/ncm-tokenizers` | Sudoku, pinned HF tokenizers and TinyStories byte BPE |
| `crates/ncm-io` | Versioned manifests, safetensors, artifact validation |
| `crates/ncm-backend-cuda` | Candle CUDA codec/operators; CPU fixture path and real-device checks |
| `crates/ncm-backend-mlx` | Native MLX codec/operators, real-Metal tests and memory probes |
| `crates/ncm-cli` | Artifact inspection and explicit-device Sudoku reconstruction/generation |
| `tools/reference` | Pinned operator/model oracles and scoped Sudoku checkpoint exporter |
| `tests/fixtures` | 18 synthetic FP32 operator cases, Sudoku tokenizer goldens and fixture policy |
| `tools/bench` | Validated aggregation of raw probe results |
| `benches` | Benchmark protocol and reproduction links |

Candle is pinned to 0.11.1 behind the backend crate's opt-in `cpu`/`cuda` features.
Default builds do not require a GPU SDK. MLX is opt-in on Apple Silicon with a
pinned native runtime. The shared backend contract supports the qualified Sudoku
codec and generator; other model profiles and sampling modes remain future work.

The CLI accepts JSON arrays of token-ID rows and requires an explicit device:

```sh
ncm inspect ARTIFACT
ncm generate ARTIFACT cuda INPUT_JSON 128
ncm reconstruct ARTIFACT mlx INPUT_JSON 128
```

Build with the corresponding `ncm-cli` feature (`cuda` on Linux, `mlx` on Apple
Silicon). Generation accepts 91-token prefixes; reconstruction accepts 89-token
boards. The final argument is an optional batch size from 1 to 128.

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
Phase 1's capability/baseline gates. Phase 2 adds qualified codec/tokenizer
parity and remediation; Phase 3 adds the scoped end-to-end Sudoku comparison.
Recheck Mac availability before starting a later phase's hardware runs.

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Research is pinned to upstream
`c70274273f489abc73832b2b98584a8483b6b203`. Model and dataset licenses must be tracked
separately; no weights or evaluation data are bundled.
