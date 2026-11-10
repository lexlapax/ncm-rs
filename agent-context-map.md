# Agent context map

Last updated: 2026-11-10 UTC. Read this first; load linked documents only as needed.

## Current state / user decisions

- Public repository: https://github.com/lexlapax/ncm-rs; `main`, remote `origin`.
- **Plan approved. Phase 1 Linux spike accepted; Apple pass next.** Within EVERY phase:
  Linux/CUDA first, then macOS/MLX; both must pass before advancing phases.
  Do not interpret this as completing the whole CUDA project before Apple work.
- Linux CPU/CUDA operators, a scoped Sudoku checkpoint exporter/checked Rust
  inspection loader, Sudoku text tokenizer and Python model traces now exist.
  **No full Rust model, inference CLI or corpus-quality result.** The general
  runtime importer and load-performance/memory qualification remain open.
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
| Scoped checkpoint exporter/loader and proposed general contract | [docs/ARTIFACTS.md](docs/ARTIFACTS.md), [ncm-io](crates/ncm-io/src/lib.rs), [exporter](tools/reference/sudoku_artifact.py) |
| Upstream tracking/upgrades | [docs/UPSTREAM.md](docs/UPSTREAM.md), [upstream.toml](upstream.toml) |
| Measured report / raw trials / reproducibility | [CUDA operator report](docs/results/2026-11-09-cuda-operators/README.md) |
| Sudoku conversion/readback/model-oracle evidence | [artifact report](docs/results/2026-11-10-sudoku-artifacts/README.md), [trace generator](tools/reference/sudoku_trace.py) |
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
  absent. Verified Sudoku-easy inputs are in ignored `artifacts/upstream/sudoku/easy`;
  final tested bundle is `artifacts/canonical/sudoku-easy-qualified` (the name does
  not imply release qualification). Clean pinned oracle checkout:
  `.reference/upstream-ncm`. Disposable `/tmp` research paths are not required.

## Previous operator/validation milestone

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

## Latest work and verification

- Added `sudoku-easy-inspection-v1`: exact input size/hash checks, weights-only
  conversion, all 238 tensors preserved (58 codec, 144 generator, 36 training).
  Canonical safetensors roundtrip and Candle CPU/CUDA upload/readback are byte-exact.
- Added bounded copied `ncm-io` loading, integrity checks and seven rejection/load
  tests. This authenticates neither the manifest nor model semantics; directory
  mutation and trusted-parent constraints are documented. No general importer yet.
- Added Rust Sudoku character codec and five pinned upstream golden cases; three
  Rust tokenizer tests include invalid IDs/characters and Python whitespace edges.
- Original vs canonical PyTorch CUDA model: 56 captures across two synthetic boards
  and all 12 levels match bit-for-bit. No upstream patches, dense uncached FP32,
  math SDPA and cloned accumulator captures. No official Sudoku quality claim.
- Public report links compact raw results, persistent/derived inventories and
  source hashes. Large 14 MiB trace archive is local/ignored; durable asset pending.
  No loader/tokenizer timing or memory claims; this candidate is not release-qualified.
- Added Hydra/OmegaConf/YAML and transitive packages to the reference Linux lock.
  `tools/check.sh` syntax-checks all reference scripts without requiring Torch;
  optional reference-venv exporter tests are invoked separately.
- Passed full `tools/check.sh --cuda`: fmt, CPU/CUDA Clippy, 15 Rust tests, docs,
  four Python aggregation tests, syntax and all 18 GPU operator fixtures. Also
  five offline exporter tests, both artifact probes and the 56-trace comparison.
  Local logs: `.reference/phase1-artifact-final-check.log`; final raw evidence:
  `reports/phase1-artifact-qualified/`. Historical operator report is unchanged.
- Previous local-validation milestone was pushed as `f7aad49`; prior operator
  milestone is `3178aa1`. No MLX execution has occurred.

## Linux Phase 1 gate closure

- Added [load/tokenizer baseline](docs/results/2026-11-10-linux-baseline/README.md):
  three fresh processes/runtime, five warmups and 30 trials. Rust/Python adapted
  canonical-load medians 201.61/241.20 ms; host RSS and global driver memory snapshots
  recorded, warm filesystem cache and background desktop GPU use disclosed.
- `torch.compile(fullgraph=True)` passes 17/18 cases; FP16 roundtrip rounding fails.
  Compiled mixed precision is explicitly unsupported; eager remains the baseline.
- Durable model traces: GitHub prerelease `phase1-fixtures-v1` at `bbdf422`.
  This is a research fixture asset, not an inference release.
- Full local CUDA validation passes. Mac availability rechecked read-only: checkout
  clean at `bbdf422`, rmlx/Ollama processes idle in sampled process listing. No MLX
  build/run yet. Linux load scope is the inspection spike, not a general importer.

## Apple candidate implementation (hardware qualification pending)

- Added target-gated mlx-rs 0.34.0 at immutable Git pin
  `adf439bf5361d8a02955de7aaf651b0c45900945` (`mlx-sys` 0.7.0, MLX 0.32.2), candidate
  composed operators, real-Metal tests, synchronous evaluation microprobe and
  checked copied unified-memory artifact/readback probe. No native Apple results
  yet; build will use a project-local metallib directory, not alter global tooling.
- Extracted the unchanged fixture schema/tolerances/comparator into
  `tests/support/operators.rs`, shared by CUDA and MLX. Full Linux `--cuda` check
  passed after extraction; unsupported non-Apple `mlx` requests fail explicitly.
- First MLX run announced after Linux gate closure. Crates.io native source build
  failed because Xcode's Metal compiler component is missing. No global tooling
  changed. The Git pin above supports prebuilt MLX; Python 3.14.10 and MLX/MLX-metal
  0.32.2 installed only in `.reference/mlx-venv`, and native include/share/lib copied
  to `.reference/mlx-native` for Python-free Rust linking. Next: rebuild using that
  native prefix and project-local rpath, then real-Metal qualification.

- Prebuilt native MLX successfully compiled/linked its bindings; Rust candidate
  needed `TryIndexOp` rather than `IndexOp` in scope (corrected). Added explicit
  local `--mlx` checks and adapted Python MLX operator/load baselines, all still
  pending real-device execution. Mac venv now also pins NumPy 2.5.4. The exact
  tested Sudoku artifact was copied to a new ignored Mac cache directory.
- Local default checks pass after these harness additions; no tolerance changes.

## Next actions (do not skip gates)

1. Announce/start the Apple Phase 1 pass: pinned mlx-rs, existing operator goldens,
   artifact readback/loading/memory, native synchronization, local checks and reports.
2. Keep compiled mixed precision and untested model features unsupported; preserve
   baseline fixtures and publish negative findings without tolerance changes.
3. Once Linux Phase 1 is ready, announce macOS/MLX work. Inspect and safely pull
   reviewed pushed commits in the Mac checkout, then port/validate that phase.
   Recheck availability if the session changes. No Mac gate is implicitly passed.
4. Only after both Phase 1 gates pass, start Phase 2 in the same staggered order.
