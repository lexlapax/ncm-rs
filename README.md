# ncm-rs

A Rust implementation of [HazyResearch/NCM](https://github.com/HazyResearch/ncm),
Next-Concept Modeling for parallel, coarse-to-fine generation. Production inference uses
native CUDA or MLX, not Python; Python is used offline for conversion, references and evaluation.

**Status: Phase 5 step 1 (`v0.1.0` release candidate) verified on Linux/CUDA and Apple Silicon/MLX; Steps 2–4 (profiling and targeted FP32 optimization) in progress.**
The unified `ncm sample` CLI, legacy aliases, feature folding, `--text`/`--decode` paths, and
fresh-clone setup guides have passed verification on both CUDA and Metal. Tagging and publishing
`v0.1.0` awaits user confirmation.
See the [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md),
[Phase 5 plan](docs/PHASE5.md), [CUDA routing/text evidence](docs/results/2026-11-19-phase5-cli/cuda-current-routing-v1.md),
[Metal step 1 evidence](docs/results/2026-11-19-phase5-cli/mlx-step1-cli-v1.md), and
[current state](agent-context-map.md).

## Profile and evidence matrix

The table describes the **Phase 3/4 qualified paths**, not a claim that every new Phase 5 CLI
option has been qualified on both devices. Results are for the specified machines, settings and
workloads; the linked reports disclose measurement limits and earlier failures.

| Profile | Qualified scope | Quality and performance evidence |
| --- | --- | --- |
| Sudoku easy, medium, hard | FP32 greedy, CUDA and Metal | [6,000-board quality and timing](docs/results/2026-11-14-phase3-sudoku/README.md); CUDA has one intermediate near-tie code difference, with no final-ID difference. |
| TinyStories | Seeded sampling, CUDA and Metal; scoped exceptions | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition); [accepted Metal numerical and latency-tail exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md). |
| XSum | Top-1, CFG 0, CUDA and Metal; scoped exceptions | [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md#per-profile-disposition); [Metal output differences](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md) and [CUDA stochastic exceptions](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md). |
| WMT14 | Scoped top-1, CUDA and Metal | [CLI and profile qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification), including quality, timing, memory and retained Metal intermediate differences. |
| LM1B small, medium, large | Seeded generation, CUDA and Metal; approved 80-seed gate | [Quality campaign](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md), [CUDA timing](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md), [Metal timing](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md#latency-and-memory-against-python-mlx), and [scoped user decisions](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md). |
| OpenWebText | Config and synthetic operators only | [Coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md); weights are unreleased and inference is refused. |

Historical results compare the tested Rust implementation with the pinned original Python CUDA
reference or a separately identified Python MLX adaptation, as appropriate. They are not benchmarks
of the current release candidate. The original LM1B five-seed entropy gate failed; the user-approved
80-seed gate and the retained failure are both documented in the linked decisions.

## CLI and current limitations

The explicit-device CLI accepts JSON arrays of token-ID rows. The new unified command dispatches on
the artifact profile; existing `sample-*` commands remain available. Run `ncm --help` or
`ncm sample --help` for current arguments and explicitly experimental options.

```sh
ncm inspect ARTIFACT
ncm sample ARTIFACT DEVICE INPUT_JSON [--seed N] [--cfg S] [--decode]
ncm sample ARTIFACT DEVICE TEXT_JSON --text [--decode]  # LM1B accepts empty strings only
ncm generate ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]     # Sudoku
ncm reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]  # Sudoku codec
```

The unified command's baseline/alias/unified equivalence passed across all nine profiles on both
CUDA ([CUDA routing](docs/results/2026-11-19-phase5-cli/cuda-current-routing-v1.md)) and Metal
([Metal step 1](docs/results/2026-11-19-phase5-cli/mlx-step1-cli-v1.md)), alongside
[text-input](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md),
[decode](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md), and
[LM1B rotary](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md) verification on both platforms. LM1B requires the separately generated,
hash-pinned rotary sidecar beside the artifact, or `--reference-rope FILE`; no table approximation
or device fallback is used. TinyStories accepts one non-padding prefix token per row; its `--text`
mode rejects strings that do not encode to exactly one non-padding token. LM1B accepts only its fixed
`[101]` start-token rows; `--text` accepts 1–32 empty strings mapped to those rows and rejects nonempty prompts.

- FP32 only in this phase; BF16/FP16 and quantization are deferred. OWT inference is unsupported.
- Neither runtime reproduces the PyTorch random stream; Metal/CUDA draw-level differences are a
  [scoped LM1B exception](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md#2-metal-stochastic-draw-differences-scoped-exception).
- XSum stochastic sampling and positive CFG, LM1B Metal `--top-p-fp32`, and WMT14's
  `mlx-retry-padded` device are experimental; they are not covered by the ordinary top-1 release bar.
  WMT14 CFG scales 0 and 1 remain unflagged; other positive scales require `--experimental-cfg-wmt14` on both unified and legacy commands. Arbitrary scales are not a blanket quality claim ([audit](docs/results/2026-11-19-phase5-cli/wmt14-cfg-policy-audit-v1.md)).
- No serving API, streaming, cancellation or concurrent-request throughput is qualified.
- Batch limits depend on model, device and available memory. On the tested 8 GiB CUDA card the
  pinned Python LM1B-large baseline ran out of memory at batch 16; its timing sweep used batch 8,
  and Rust large batch 16 was not measured ([report](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md)).
  CLI argument limits are not a guarantee that a batch fits.
- The [setup guide](docs/USAGE.md) was verified from fresh clones on both Linux
  ([Linux replay](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md)) and macOS
  ([macOS replay](docs/results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md)). The historical Sudoku
  medium/hard exporter identity is unresolved; provenance-only re-exports are documented in the
  [export investigation](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md).

Release notes are prepared in [docs/RELEASE-NOTES-v0.1.0.md](docs/RELEASE-NOTES-v0.1.0.md) (awaiting tag/release approval). No model weights,
evaluation data or prebuilt inference binaries are included in Git.

## Build and local checks

Rust **1.99.1+** (edition 2024), rustfmt, Clippy and Python 3 are required for local checks. The
normal workspace does not require a GPU SDK. CUDA builds require a compatible toolkit, driver,
compute capability and cuDNN for `cuda-cudnn`; MLX builds require Apple Silicon and matching native
MLX libraries. The tested development setups are documented in the
[reference tools guide](tools/reference/README.md); they are not yet a fresh-clone release recipe.

```sh
./tools/check.sh
# Linux, tested RTX 3060 Ti configuration; real GPU required:
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
# Apple Silicon with the documented native MLX environment:
./tools/check.sh --mlx
```

Checks do not download weights or run SSH jobs. Default checks exercise CPU fixtures and report
utilities; opt-in GPU checks run on real hardware. Passing an operator smoke test is not a model
quality or performance gate. Build CI is intentionally disabled; the metadata-only upstream watcher
remains enabled.

## Workspace and project documents

| Path | Responsibility |
| --- | --- |
| `crates/ncm-core`, `crates/ncm-model` | Validated configuration, shared FP32 model and backend contracts |
| `crates/ncm-tokenizers`, `crates/ncm-io` | Pinned tokenizers and checked artifact loading |
| `crates/ncm-backend-cuda`, `crates/ncm-backend-mlx` | Explicit CUDA and native MLX adapters |
| `crates/ncm-cli` | Inspection and explicit-device inference CLI |
| `tools/reference`, `tests/fixtures`, `tools/bench` | Offline conversion, pinned oracles, tests and reporting |

See [architecture](docs/ARCHITECTURE.md), [plan](docs/PLAN.md),
[testing](docs/TESTING.md), [artifact contract](docs/ARTIFACTS.md),
[reporting](docs/REPORTING.md) and [upstream policy](docs/UPSTREAM.md). Agents start with
[AGENTS.md](AGENTS.md) and the [context map](agent-context-map.md).

## Provenance and license

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The pinned upstream code is
`c70274273f489abc73832b2b98584a8483b6b203`; checkpoint revision is
`d342ce8aaea58830eef3b8ad473664673e41f692`. Model and dataset licenses are tracked separately.
