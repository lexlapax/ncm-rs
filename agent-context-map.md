# Agent context map

Last updated: 2026-11-10 UTC. Read this first; load linked documents only as needed.

## Current state / user decisions

- Public repository: https://github.com/lexlapax/ncm-rs; `main`, remote `origin`.
- **Plan approved. Phase 1 in progress, Linux portion.** Within EVERY phase:
  Linux/CUDA first, then macOS/MLX; both must pass before advancing phases.
  Do not interpret this as completing the whole CUDA project before Apple work.
- Linux CPU/CUDA candidate operators and synthetic fixtures now exist. There is
  **no full model, checkpoint converter, inference CLI, or model-quality result**.
- `ssh macstudio` is authorized and the user has now confirmed it is **available
  for MLX**. Announce the first run when the Linux phase gate is ready; recheck
  availability if resuming in a later session. Only read-only inspection has
  occurred there; no Apple code/build/benchmark has been launched.
- The user requested replacing GitHub build CI with local validation. Use
  `tools/check.sh` and opt-in `--cuda`; do not restore build CI without approval.
  The metadata-only weekly upstream watcher remains on GitHub.
- Mac checkout exists, clean at initial inspection: `~/projects/lexlapax/ncm-rs`.
  Commit/push locally as requested, then use a safe pull there when appropriate;
  inspect remote worktree first, never overwrite user work or force-push.
- Continue periodic upstream parity and public quality/performance reporting.
  Update this map before EVERY commit. No agent attribution or co-author trailers.

## Task routing

| Need | Load |
| --- | --- |
| Goals/layout/build commands | [README.md](README.md), [Cargo.toml](Cargo.toml) |
| Current milestone and phase gates | [docs/PLAN.md](docs/PLAN.md) |
| NCM semantics, release configs, source evidence/bugs | [docs/RESEARCH.md](docs/RESEARCH.md) |
| Architecture and provisional runtime contract | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Candidate tensor operations / opt-in features | [CUDA manifest](crates/ncm-backend-cuda/Cargo.toml), [ops.rs](crates/ncm-backend-cuda/src/ops.rs) |
| Fixtures and regression tests | [test policy](docs/TESTING.md), [fixture README](tests/fixtures/README.md), [operators.rs](crates/ncm-backend-cuda/tests/operators.rs) |
| Oracle and Python environment | [tools/reference/README.md](tools/reference/README.md) |
| Checkpoint format/security (not implemented) | [docs/ARTIFACTS.md](docs/ARTIFACTS.md) |
| Upstream tracking/upgrades | [docs/UPSTREAM.md](docs/UPSTREAM.md), [upstream.toml](upstream.toml) |
| Measured report / raw trials / reproducibility | [CUDA operator report](docs/results/2026-11-09-cuda-operators/README.md) |
| Publication policy and report tools | [docs/REPORTING.md](docs/REPORTING.md), [compare.py](tools/bench/compare.py) |
| Agent rules / local checks | [AGENTS.md](AGENTS.md), [tools/check.sh](tools/check.sh) |
| Weekly metadata-only upstream watcher | [.github/workflows/upstream-watch.yml](.github/workflows/upstream-watch.yml) |

## Pins / machines / important limitations

- NCM research pin `c70274273f489abc73832b2b98584a8483b6b203`; HF inventory
  `d342ce8aaea58830eef3b8ad473664673e41f692`. Not model-level Rust compatibility.
- Candle 0.11.1, opt-in backend features `cpu` and `cuda`. Default build needs no
  CUDA SDK. Model/shared backend abstraction and MLX integration remain provisional.
- Supported Rust floor is now **1.99.1**, the tested conservative version, not
  an assertion that all older releases fail. Candle 0.11.1 failed on the former
  1.88 claim (`stdarch_x86_avx512`, target features, `__cpuid_count`); do not
  restore that unverified support claim.
- Linux: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA 13.4, Rust 1.99.1.
  Successful release build env: `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86
  CARGO_BUILD_JOBS=4`. System default GCC is newer; keep the tested host compiler.
- Offline Python venv `.reference/venv`: Python 3.14.9, Torch 2.15.0+cu132.
  Full Linux lock is `tools/reference/requirements-linux.lock`; not suitable for Mac.
- Mac: M1 Ultra 64 GiB, macOS 26.7.3, Xcode 27.2, Homebrew Rust 1.99.1.
  Noninteractive SSH needs `PATH=/opt/homebrew/bin:$PATH` for Cargo/uv.
- OWT weights absent at inspected HF revision; full upstream training pipeline
  absent. No checkpoints downloaded yet. Research clones in `/tmp` are disposable;
  do not make tools depend on their continued existence.

## Latest work and verification

- 18 synthetic operator fixtures: matmul, conv/transposed-conv layouts (including
  prime stride 89), group/layer/RMS norms, uneven/edge area and linear interpolation,
  padded prefix attention, split-half RoPE resets and nontrivial FP16 rounding.
- All 18 pass Candle CPU and real CUDA against fixed PyTorch CPU reference at
  `atol=1e-5, rtol=1e-4`. PyTorch CUDA also passes. Three fresh-process timing runs
  per runtime, five warmups and thirty trials/case/run, published with raw data.
- Negative result: composed interpolation is much slower than PyTorch kernels.
  No generation speedup, model quality or peak-memory claim. Optimization deferred.
- Passed: workspace CPU-feature tests (5 Rust tests), CPU-feature Clippy with
  warnings denied, formatting, CPU-feature rustdoc with warnings denied,
  4 Python report-tool tests, Python compilation, local links/TOML validation,
  measured-source SHA verification and whitespace checks.
- Replaced `.github/workflows/ci.yml` with executable `tools/check.sh`: fmt,
  default check, CPU-feature Clippy/tests/docs, Python report-tool tests/syntax;
  opt-in `--cuda` adds CUDA Clippy plus a real-device release probe. No implicit
  downloads, SSH, GPU fallback or tool installation. ShellCheck runs if present.
- Passed script validation: `bash -n`, ShellCheck, help, unknown-argument exit 2,
  complete default run invoked from `/tmp`, and complete `--cuda` run with the
  tested compiler/architecture env. All 18 GPU cases passed; smoke output is
  `reports/local-check-20261110T003546Z-2451498/cuda.json` (ignored local artifact).
- This change affects validation orchestration/toolchain metadata only, not model
  or operator execution. Prior public measurements remain historical and unchanged;
  no new generation-quality/performance claim or full benchmark comparison.

## Next actions (do not skip gates)

1. Commit/push the local-validation change with this context update. The previous
   CUDA operator milestone is already pushed as `3178aa1`.
2. Continue Phase 1 Linux: trusted pinned checkpoint conversion, tensor inventory,
   canonical artifact roundtrip, tokenizer IDs and full model-level oracle traces.
   The synthetic operator suite is only a subset of Phase 1 acceptance.
3. Once Linux Phase 1 is ready, announce macOS/MLX work (user reports availability
   now). Inspect and safely pull reviewed commits in the Mac checkout, then
   port/validate that phase there. Recheck availability if the session changes.
4. Only after both Phase 1 gates pass, start Phase 2 in the same staggered order.
