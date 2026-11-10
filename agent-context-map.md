# Agent context map

Last updated: 2026-11-10 UTC. Read this first; load linked documents only as needed.

## Current state / user decisions

- Public repository: https://github.com/lexlapax/ncm-rs; `main`, remote `origin`.
- **Plan approved. Phase 1 in progress, Linux portion.** Within EVERY phase:
  Linux/CUDA first, then macOS/MLX; both must pass before advancing phases.
  Do not interpret this as completing the whole CUDA project before Apple work.
- Linux CPU/CUDA candidate operators and synthetic fixtures now exist. There is
  **no full model, checkpoint converter, inference CLI, or model-quality result**.
- `ssh macstudio` is authorized. It has competing workloads: **notify the user
  and coordinate availability BEFORE starting MLX workloads**. Only read-only
  inspection has occurred there; no Apple code/build/benchmark was launched.
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
| Agent rules / CI | [AGENTS.md](AGENTS.md), [.github/workflows/ci.yml](.github/workflows/ci.yml) |

## Pins / machines / important limitations

- NCM research pin `c70274273f489abc73832b2b98584a8483b6b203`; HF inventory
  `d342ce8aaea58830eef3b8ad473664673e41f692`. Not model-level Rust compatibility.
- Candle 0.11.1, opt-in backend features `cpu` and `cuda`. Default build needs no
  CUDA SDK. Model/shared backend abstraction and MLX integration remain provisional.
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
- CI expanded to CPU fixture tests on declared MSRV 1.88 and stable, plus report
  tooling. Confirm new remote run after push; local checks used Rust 1.99.1.

## Next actions (do not skip gates)

1. Commit/push this operator milestone and public report as requested; this map
   is updated for that commit. No further Apple activity until coordination.
2. Continue Phase 1 Linux: trusted pinned checkpoint conversion, tensor inventory,
   canonical artifact roundtrip, tokenizer IDs and full model-level oracle traces.
   The synthetic operator suite is only a subset of Phase 1 acceptance.
3. Once Linux Phase 1 is ready, notify user before embarking on macOS/MLX. Pull the
   reviewed commits into the Mac checkout, then port/validate that phase there.
4. Only after both Phase 1 gates pass, start Phase 2 in the same staggered order.
