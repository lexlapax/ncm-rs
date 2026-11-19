# ncm-rs

A Rust implementation of [HazyResearch/NCM](https://github.com/HazyResearch/ncm),
Next-Concept Modeling for parallel, coarse-to-fine generation. Production inference uses
native CUDA or MLX, not Python; Python is used offline for conversion, references and evaluation.

**Status: Phase 4 complete with scoped, user-accepted exceptions; Phase 5 release candidate in progress.**
This is not yet a released inference package. The Phase 5 unified CLI has CUDA development screens,
but its Metal validation, fresh-clone setup, performance evidence and release gates remain pending.
See the [Phase 4 audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md),
[Phase 5 plan](docs/PHASE5.md) and [current state](agent-context-map.md).

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
ncm sample ARTIFACT DEVICE TEXT_JSON --text [--decode]  # TinyStories, XSum or WMT14
ncm generate ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]     # Sudoku
ncm reconstruct ARTIFACT DEVICE INPUT_JSON [BATCH_SIZE]  # Sudoku codec
```

The unified command's CUDA [baseline/alias equivalence screen](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md)
covers one fixed batch per profile. Its [text-input](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md),
[decode](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md) and
[LM1B rotary](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md) screens have narrower scopes.
Metal equivalents are pending. LM1B requires the separately generated, hash-pinned rotary sidecar
beside the artifact, or `--reference-rope FILE`; no table approximation or device fallback is used.
TinyStories accepts one non-padding prefix token per row; its `--text` mode rejects strings
that do not encode to exactly one non-padding token. LM1B accepts only its fixed `[101]`
start-token rows and currently rejects `--text`.

- FP32 only in this phase; BF16/FP16 and quantization are deferred. OWT inference is unsupported.
- Neither runtime reproduces the PyTorch random stream; Metal/CUDA draw-level differences are a
  [scoped LM1B exception](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md#2-metal-stochastic-draw-differences-scoped-exception).
- XSum stochastic sampling and positive CFG, LM1B Metal `--top-p-fp32`, and WMT14's
  `mlx-retry-padded` device are experimental; they are not covered by the ordinary top-1 release bar.
- No serving API, streaming, cancellation or concurrent-request throughput is qualified.
- Batch limits depend on model, device and available memory. On the tested 8 GiB CUDA card the
  pinned Python LM1B-large baseline ran out of memory at batch 16; its timing sweep used batch 8,
  and Rust large batch 16 was not measured ([report](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md)).
  CLI argument limits are not a guarantee that a batch fits.
- The [setup guide](docs/USAGE.md) is an incomplete draft; fresh-clone verification is pending. In
  particular, the historical Sudoku medium/hard exporter identity is unresolved; see the
  [export investigation](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md).

Release notes are a [draft](docs/RELEASE-NOTES-v0.1.0.md), not a published release. No model weights,
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
