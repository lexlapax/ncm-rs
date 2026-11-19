# Agent context map

Last updated: 2026-11-18 UTC. Read first; load linked context lazily. Keep this file short:
current state, decisions, blockers and next actions. Put durable evidence in `docs/results/`.
The full pre-compaction history is archived in [docs/history/context-map-2026-11-18.md](docs/history/context-map-2026-11-18.md).

## Current state

- **Phases 0–3 complete.** Phase 2 closed with a [scoped Apple immediate-memory exception](docs/results/2026-11-11-phase2-optimization/phase2-final-audit-v1.md).
  Phase 3 Sudoku is complete on CUDA and Metal ([report](docs/results/2026-11-14-phase3-sudoku/README.md)).
- **Phase 4 active** ([plan](docs/PLAN.md#phase-4--released-text-profile-compatibility), [contract](docs/PHASE4-TEXT.md), [handoff](RESUME.md)).

| Profile | Status |
| --- | --- |
| TinyStories | Advanced under [user-accepted exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md) |
| XSum | Advanced under the [Metal](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md) and [CUDA stochastic](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md) exceptions |
| WMT14 | Scoped CUDA and Metal top-1 qualification ([evidence](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification)) |
| LM1B-small | **Active.** Quality gate and Metal performance open (below) |
| LM1B-medium/large | Checkpoint inventories only; model code is still small-specific (`crates/ncm-model/src/lm1b_small.rs`) |
| OWT | Synthetic/config/operator coverage required; none found in `crates/`. Weights unreleased |

- **LM1B-small evidence:** CUDA fixed-history traces, greedy and CLI checks pass. Repeated CUDA
  timing beats pinned Python in median, p95 and process RSS at batches 1, 4 and 16; Rust model load
  is slower ([performance](docs/results/2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements)).
  The frozen five-seed entropy interval failed on CUDA and every observed-seed retest; the
  failures stay on record ([quality](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md)).
  Metal fixed-history decisions and IDs match CUDA; draw-level differences are now covered by
  the exception below. Metal quality and Metal performance against Python MLX are not yet measured.
  The independent [Python MLX reference](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#metal-development-reference-for-complete-requests)
  is `tools/reference/lm1b_small_mlx_reference.py`.

## User decisions (2026-11-18)

Recorded in [user-decisions-v1.md](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md):

1. The fresh **80-seed quality protocol** is the LM1B gate (same scorer, margins and bootstrap).
   Seeds are pre-registered in [cohort-seeds-v1.json](docs/results/2026-11-16-phase4-lm1b/cohort-seeds-v1.json)
   (SHA-256 `c9800090de18f18d6719993f1ac36057b6cea479a26cc2e4b3115de6e4beeab1`). Do not change them after generation starts.
2. **Stop Metal-vs-CUDA draw-level parity work.** Those differences are a scoped exception; fixed-history
   decision/ID differences remain failures.
3. **Medium/large bar:** export plus identity checks, teacher-forced decision/ID parity at 1–2 shapes,
   CLI generation, 80-seed quality, one batch-sweep latency/memory comparison per device. No new
   numerical deep-dives unless one of these fails.
4. The map was compacted; history is archived.

## Next actions, in order

1. Promote the 80-seed campaign tooling from ignored `reports/phase4-lm1b-cohort-campaign-v1/` into
   tracked tools. Run the prospective CUDA campaign (Python and Rust CUDA lists) and publish it.
2. Metal: generate the `rust_metal` list and compare it with the same Python CUDA captures. Run a
   Metal batch 1/4/16 latency/memory comparison against the Python MLX reference. Then close LM1B-small.
3. Generalize LM1B model/config/CLI from small to medium, then large; apply the bar above.
4. Add OWT synthetic/config/operator coverage.
5. Audit Phase 4 against the plan and contract, update docs, and mark Phase 4 complete.

## Working rules carried forward

- Linux/CUDA first, then Metal, for each step. `ssh macstudio` is authorized; inspect the clean
  remote checkout and fast-forward pull pushed commits; do not reboot the Mac or interrupt `dotsetup.rmlx`.
- Keep `.pi/` untouched. Ignored `reports/` hold raw captures; tracked docs hold identities and conclusions.
- CUDA check: `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.

## Checks run for this update

- Docs-only change: the map was compacted, user decisions and pre-registered seeds were added, and RESUME.md was updated.
  All relative links in the archive and in this map were verified to resolve.

## Task routing

| Need | Load |
| --- | --- |
| Phase 2 contract / acceptance / remediation | [final dual-platform acceptance](docs/results/2026-11-10-phase2-acceptance/README.md), [docs/PHASE2.md](docs/PHASE2.md), [plan](docs/PLAN.md), [research and performance extension](docs/PHASE2-PERFORMANCE.md), [initial research/PoC](docs/results/2026-11-10-phase2-performance-research/README.md) |
| Approved additional Phase 2 optimization | [five-step plan, research and pending evidence checklist](docs/PHASE2-OPTIMIZATION-REVIEW.md) (implementation authorized; no new speedup result) |
| Shared runtime and model | [core](crates/ncm-core/src/lib.rs), [config](crates/ncm-core/src/config.rs), [model](crates/ncm-model/src/lib.rs), [blocks](crates/ncm-model/src/blocks.rs), [codec](crates/ncm-model/src/codec.rs) |
| CUDA adapter / known native fixes | [adapter](crates/ncm-backend-cuda/src/codec.rs), [vectorized resampling](crates/ncm-backend-cuda/src/resample.rs), [pool policy](crates/ncm-backend-cuda/src/runtime.rs), [operators](crates/ncm-backend-cuda/src/ops.rs), [remediation](docs/results/2026-11-10-phase2-cuda-remediation/README.md) |
| Phase 2 oracle and complete CUDA results | [oracle audit](docs/results/2026-11-10-phase2-oracle/README.md), [codec report](docs/results/2026-11-10-phase2-codec/README.md) |
| Tokenizer identities, safety and parity | [tokenizer report](docs/results/2026-11-10-phase2-tokenizers/README.md), [source lock](tools/reference/tokenizer-sources.lock.json), [canonical lock](tools/reference/tokenizers-v1.lock.json) |
| Fixtures / offline tools | [fixtures](tests/fixtures/README.md), [reference README](tools/reference/README.md) |
| Artifact trust/limits | [docs/ARTIFACTS.md](docs/ARTIFACTS.md), [ncm-io](crates/ncm-io/src/lib.rs), [runtime checks](docs/results/2026-11-10-phase2-runtime/README.md) |
| Architecture / upstream caveats | [architecture](docs/ARCHITECTURE.md), [research](docs/RESEARCH.md) |
| Checks / reporting / compatibility | [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md) |
| MLX codec / current parity evidence | [adapter](crates/ncm-backend-mlx/src/codec.rs), [Apple parity report](docs/results/2026-11-10-phase2-mlx-codec/README.md), [shared trained capture](tests/support/trained_codec.rs) |
| Phase 2 Linux acceptance / benchmarks | [Linux gate](docs/results/2026-11-10-phase2-linux/README.md), [summary tool](tools/bench/phase2_summary.py), [runner](tools/bench/run_phase2_linux.py) |
| Historical baselines | [results index](docs/results/README.md), [Apple Phase 1](docs/results/2026-11-10-mlx-phase1/README.md) |

## Pins, machines and caches

- Upstream NCM: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF checkpoints: `hazyresearch/ncm-checkpoints` at
  `d342ce8aaea58830eef3b8ad473664673e41f692`. OWT weights remain absent.
- Candle `=0.11.1`; default workspace needs no GPU SDK. Explicit CPU diagnostic
  and CUDA paths never silently fall back. Rust floor **1.99.1** is tested;
  previous 1.88 failed Candle AVX-512 intrinsics, not all intermediate versions.
- Linux: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA 13.4. Command:
  `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.
  Default GCC 16 is not the tested CUDA host compiler.
- Linux Torch reference: `.reference/venv`, Python 3.14.9, Torch 2.15.0+cu132;
  `tools/reference/requirements-linux.lock`. Clean source: `.reference/upstream-ncm`.
- Sudoku source: `artifacts/upstream/sudoku/easy`; canonical inspected bundle on
  both machines: `artifacts/canonical/sudoku-easy-qualified`. Its name alone does
  not confer inference qualification. Original file/tensor hashes are committed
  under `docs/results/2026-11-10-sudoku-artifacts/raw/`.
- Tokenizer source: `artifacts/upstream/tokenizers-v2`; canonical:
  `artifacts/canonical/tokenizers-qualified`. Separate `.reference/tokenizers-venv`
  uses `requirements-tokenizers.lock` (Transformers 4.57.6, Tokenizers 0.22.2,
  regex 2025.9.18 / Unicode 16). Do not replace the Torch environment with it.
- Rust HF Tokenizers `=0.22.2`, only Oniguruma feature; no HTTP/remote code. Candle's
  pre-existing Tokenizers 0.23.2 remains separate. Tiny byte BPE uses locked
  fancy-regex 0.17.0 and Unicode-16 regex-syntax tables. Asset licenses are retained.
- Mac: M1 Ultra 64 GiB, macOS 26.7.3, Xcode 27.2, Rust 1.99.1. Noninteractive SSH:
  `export PATH=/opt/homebrew/bin:$PATH`. Optional Xcode Metal compiler is missing;
  do not change global Xcode. Tested project-local prebuilt native MLX route works.
- MLX bindings: mlx-rs 0.34.0 / mlx-sys 0.7.0 Git
  `adf439bf5361d8a02955de7aaf651b0c45900945`; native MLX 0.32.2, mlx-c
  `a341b4925024b88b2c593468f16e12f5e17315da`. `.reference/mlx-venv` is offline-only;
  native include/share/lib are copied into `.reference/mlx-native`.
- Mac candidate testing used a separate worktree `.reference/phase2-mlx-candidate`
  with the main checkout's native prefix and target cache. Main checkout is updated
  only by pulling pushed commits. Preserve unrelated work; existing CPU work was
  left running, builds limited to two jobs. Benchmark acceptance must assess noise.
- Mac build: `MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"`,
  `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"`,
  `RUSTDOCFLAGS="$RUSTFLAGS"`, `CARGO_BUILD_JOBS=8`, then `./tools/check.sh --mlx`.
  Tested wheel needs macOS 26.2+; absolute rpath is a spike setup, not packaging.

