# Agent context map

Last updated: 2026-11-10 UTC. Read first; load linked context lazily.

## Current state and user decisions

- Public repository: https://github.com/lexlapax/ncm-rs, `main`, remote `origin`.
- **User goal: complete Phase 1. Both gates now passed, Linux/CUDA first and
  macOS/MLX second. Phase 2 has not started.** See [acceptance](docs/PHASE1.md).
  Keep this same platform order within EVERY later phase; both gates before advance.
- Implemented: candidate CPU/CUDA/MLX operators, shared fixed fixture contract,
  scoped Sudoku exporter/checked copied loader, Sudoku text tokenizer, upstream
  model traces and reproducible operator/load/tokenizer/memory reports.
- **No full Rust codec/generator, inference CLI, official task-quality result,
  generalized runtime importer or inference release.** Core/model crates remain
  scaffolds. Phase 1 completion is feasibility/reference qualification only.
- Production inference stays Python-free. Offline conversion/oracles and native
  CUDA/MLX libraries are allowed. Never silently relax tolerances or advance pins.
- Local checks replace GitHub build CI by user request. Use `tools/check.sh`,
  explicit `--cuda` or `--mlx`; never restore build CI without approval. Only the
  metadata-only upstream watcher remains on GitHub.
- `ssh macstudio` is authorized. First MLX run was announced and completed after
  Linux acceptance. Recheck availability before later hardware runs; idle inference
  services were left untouched. Inspect worktree and safely pull reviewed pushed
  commits into `~/projects/lexlapax/ncm-rs`; never overwrite unrelated work.
- Update this map before EVERY commit, including amend/fixup. No agent attribution
  or co-author trailers. Preserve required upstream licensing/provenance.

## Task routing

| Need | Load |
| --- | --- |
| Completed gate and precise scope | [docs/PHASE1.md](docs/PHASE1.md) |
| Next stages | [docs/PLAN.md](docs/PLAN.md) |
| Architecture / provisional model interface | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Upstream semantics and known bugs | [docs/RESEARCH.md](docs/RESEARCH.md) |
| Artifact validation/security limits | [docs/ARTIFACTS.md](docs/ARTIFACTS.md), [ncm-io](crates/ncm-io/src/lib.rs) |
| Oracle/export/build reproduction | [tools/reference/README.md](tools/reference/README.md) |
| Shared golden schema and numerical tolerance | [tests/support/operators.rs](tests/support/operators.rs), [test policy](docs/TESTING.md) |
| CUDA / MLX compositions | [CUDA ops](crates/ncm-backend-cuda/src/ops.rs), [MLX ops](crates/ncm-backend-mlx/src/ops.rs) |
| Published evidence, including negative findings | [results index](docs/results/README.md), [Apple report](docs/results/2026-11-10-mlx-phase1/README.md) |
| Reporting and upstream qualification | [docs/REPORTING.md](docs/REPORTING.md), [docs/UPSTREAM.md](docs/UPSTREAM.md), [upstream.toml](upstream.toml) |
| Local checks / agent constraints | [tools/check.sh](tools/check.sh), [AGENTS.md](AGENTS.md) |

## Pins, machines and caches

- Upstream NCM: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF `hazyresearch/ncm-checkpoints`: `d342ce8aaea58830eef3b8ad473664673e41f692`.
- Candle 0.11.1; optional `cpu`/`cuda`. Default builds need no GPU SDK.
- mlx-rs 0.34.0 / mlx-sys 0.7.0 at immutable Git pin
  `adf439bf5361d8a02955de7aaf651b0c45900945`, native MLX 0.32.2. Apple-only `mlx` feature;
  non-Apple requests error, never fall back to CPU or Candle Metal.
- Rust floor 1.99.1 is tested, conservative. Previous 1.88 failed Candle AVX-512
  intrinsics; no claim that every intermediate release fails.
- Linux: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA 13.4, Rust 1.99.1. Tested command:
  `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.
  System GCC 16 is not the tested CUDA host compiler.
- Linux offline venv `.reference/venv`: Python 3.14.9, Torch 2.15.0+cu132;
  `tools/reference/requirements-linux.lock` is not suitable for Mac.
- Mac: M1 Ultra 64 GiB, macOS 26.7.3, Xcode 27.2, Rust 1.99.1. Noninteractive SSH:
  `export PATH=/opt/homebrew/bin:$PATH`. Xcode lacks the optional Metal compiler.
  No global tooling changed: `.reference/mlx-venv` holds Python 3.14.10, MLX/MLX-metal
  0.32.2, NumPy 2.5.4; native include/share/lib copied to `.reference/mlx-native`.
  Tested prebuilt wheel requires macOS 26.2+, older tuples unqualified.
- Mac build/check env: `MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"`,
  `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"`,
  `RUSTDOCFLAGS="$RUSTFLAGS"`, `CARGO_BUILD_JOBS=8`, then `./tools/check.sh --mlx`.
  Rust runtime needs only copied native libraries, not the Python venv.
- Linux inputs: `artifacts/upstream/sudoku/easy`; tested canonical bundle on both
  machines: `artifacts/canonical/sudoku-easy-qualified` (inspection profile only).
  Clean pinned oracle checkout on both: `.reference/upstream-ncm`. All ignored.
- Durable 56-trace archive: GitHub research prerelease `phase1-fixtures-v1` at
  `bbdf422`; SHA-256 in original oracle report. Not an inference release.

## Verified results and remaining limitations

- All 18 fixed FP32 operator cases pass Candle CPU, CUDA, Rust MLX and adapted
  Python MLX at unchanged `atol=1e-5, rtol=1e-4`. Largest MLX error ~4.77e-7.
- All 238 source tensors preserved (58 codec, 144 generator, 36 training), exact
  safetensors roundtrip plus exact CPU/CUDA/MLX readback. Original vs converted
  upstream Python CUDA yields 56 exact traces across two synthetic boards/12 levels.
- Five tokenizer goldens pass on both architectures. Rust input rejection tests,
  seven artifact/security tests and two real-Metal tests pass. Loader checks
  integrity, not manifest authenticity/model semantics; trusted directory/no
  concurrent mutation required. Payload byte bounds are NOT peak-RSS guarantees.
- Local Linux `--cuda` and Mac `--mlx` pass; Mac covers 15 base Rust tests plus two
  Metal tests; seven Python report-tool tests pass. Five optional Torch exporter
  tests also passed again during final Linux closure. No full model tests implied.
- Three fresh-process runs/runtime, five warmups and thirty trials published.
  Linux Rust/Python copied-load medians ~202/241 ms. Apple ~939/282 ms: Rust slower.
  Apple repeated-load host peak RSS ~4.40 GiB Rust vs ~0.41 GiB Python; investigate
  before production memory-efficiency claims. MLX active arrays return to zero,
  allocator caches retain ~155 MiB; counters are not total host RSS or CUDA VRAM.
- Both Apple load preflights execute a loaded-weight Metal kernel. Earlier
  Python leaf-only observations were superseded before publication. Rust probe
  also passes with a cleared environment/no reference venv; no Python linkage.
- Candle composed interpolation remains slower than PyTorch. Torch fullgraph
  compilation passes 17/18 cases but loses FP16-roundtrip rounding: compiled mixed
  precision unsupported, no tolerance waiver. Eager FP32 remains the baseline.
- No task-quality or model-generation speedup claim. Workstation noise/warm disk
  caches disclosed. Generalized configs, TinyStories assets, mixed-precision models,
  cached generation, packaging and training remain future phases. OWT weights and
  the upstream full training pipeline remain unavailable.

## Latest provenance and next actions

- CUDA milestone `3178aa1`; local checks `f7aad49`; artifacts/traces `bbdf422`;
  Linux load gate `e8b81c4`; MLX candidates `ee5f51a` / `8f18246`; measured operator/
  Rust load sources `2d262e1`; matched Python load preflight/aggregation `2b19c87`.
- Final Apple raw trials: `reports/phase1-final/mlx` and `mlx-load-matched` locally;
  remote `reports/phase1-mlx/final` and `load-final`. Final remote local-check log:
  `reports/phase1-mlx/check-closure.log`; Linux `.reference/phase1-linux-closure.log`.
- Phase 1 goal is complete. Next plan stage is Phase 2 shared primitives/codec,
  starting on Linux and only then Mac. Preserve golden/reference behavior; retain
  aliasing-helper caveats and copied accumulator snapshots. Do not begin claiming
  model compatibility from the operator/artifact evidence or hide negative baselines.
