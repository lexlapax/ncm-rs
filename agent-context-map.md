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
| LM1B-small | **Active.** 80-seed quality passes on CUDA and Metal; CUDA timing passes; Metal latency passes, Metal RSS fails at batches 4/16 (decision pending) |
| LM1B-medium/large | Artifacts done; model generalized by size; CUDA teacher-forced parity passes on both backends ([parity](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-parity-v1.md)). Next: 80-seed CUDA quality and timing, then Metal |
| OWT | Synthetic/config/operator coverage required; none found in `crates/`. Weights unreleased |

- **LM1B-small evidence:** CUDA fixed-history traces, greedy and CLI checks pass. Repeated CUDA
  timing beats pinned Python in median, p95 and process RSS at batches 1, 4 and 16; Rust model load
  is slower ([performance](docs/results/2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements)).
  The frozen five-seed entropy interval failed on CUDA and every observed-seed retest; the
  failures stay on record ([quality](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md)).
  The approved **80-seed CUDA gate passes**: perplexity +0.44 [-0.27, 1.15], entropy +0.00073
  [-0.00032, 0.00180] ([result](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-result)).
  Metal fixed-history decisions and IDs match CUDA; draw-level differences are now covered by
  the exception below. Metal quality and Metal performance against Python MLX are not yet measured.
  The independent [Python MLX reference](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#metal-development-reference-for-complete-requests)
  is `tools/reference/lm1b_small_mlx_reference.py`.

- **Mac on hold (2026-11-19):** the user is rebooting macstudio. Do not run anything there until the
  user says it is ready; then recheck its state. Pending Mac step: copy `reports/phase4-lm1b-mlx-timing-v1/` to Linux.
- **Metal results:** the 80-seed gate passes, with perplexity +0.29 [-0.41, 0.97] and entropy +0.00103
  [-0.00007, 0.00213] ([result](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-metal-result)).
  The full timing run is faster at batches 1/4/16 (13/20/35% at the median, also at p95). RSS passes at
  batch 1 but fails at 4 (+1.4%) and 16 (+12%); Python RSS is flat at 1,681 MiB
  ([timing](docs/results/2026-11-16-phase4-lm1b/small-metal-timing-v1.md)).

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

1. Done except one decision: Metal RSS at batches 4/16 (accept a scoped exception, or reduce host
   logits residency). Earlier text in this step: generate the `rust_metal` list (device FP32 sampler) on the Mac and score it against the same Python CUDA captures from
   `reports/phase4-lm1b-campaign-prospective-cuda-v1/`. Run a Metal batch 1/4/16 latency/memory
   comparison against the Python MLX reference. Then close LM1B-small.
   Run checks under `setpriv --inh-caps=-all`; the shell may carry an inheritable capability.
2. Generalize LM1B model/config/CLI from small to medium, then large; apply the bar above.
3. Add OWT synthetic/config/operator coverage.
4. Audit Phase 4 against the plan and contract, update docs, and mark Phase 4 complete.

## Working rules carried forward

- Linux/CUDA first, then Metal, for each step. `ssh macstudio` is authorized; inspect the clean
  remote checkout and fast-forward pull pushed commits; do not reboot the Mac or interrupt `dotsetup.rmlx`.
- Keep `.pi/` untouched. Ignored `reports/` hold raw captures; tracked docs hold identities and conclusions.
- CUDA check: `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.

## Checks run for this update

- **LM1B medium/large step 2 (CUDA parity):** `Lm1bSize` (profile, config hash, layers, heads, width) drives
  the teacher, CLI (`ncm sample-lm1b`), and CUDA whitelists (efficient attention heads 16/20, fused rotary
  12/16/20). Both CUDA backends pass 51/51 teacher-forced tensors with zero argmax and ID mismatches
  at all three sizes against matching references: the cuDNN backend against strict, and the efficient backend
  against evaluator settings (default attention dispatch, convolution TF32). The efficient path against strict fails by
  design: level-zero reconstruction equals PyTorch with convolution TF32 exactly.
  `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes. Campaign size support is in the working tree (untested).
- **LM1B medium/large step 1:** the exporter takes `--size`; medium and large artifacts (manifests `2fea4a25…`,
  `86676fed…`) are deterministic across two exports, and `ncm-io` has separate per-size budgets plus a new test.
  The generalized CPU probe loads all three sizes. `setpriv --inh-caps=-all ./tools/check.sh` passes.
  Next: make the teacher read layers, heads and width from config; audit CUDA fused kernels for width
  assumptions; extend the trace tools, campaign and CLI by size.
- **Metal RSS fix attempt (user chose "try one targeted fix"; Metal untested, Mac on hold):** host FP64
  penalties now read decoder logits through `visit_f32_rows` in blocks of two rows
  (`LM1B_PENALTY_ROWS`). On MLX each block is a separate exact GPU copy (multiply by one), so the CPU
  never touches the full logits buffer. The default (CUDA/CPU) implementation slices one host copy.
  Values and order are unchanged. Hypothesis being tested: Rust RSS growth equals the CPU-touched
  decoder logits. Checks: a new CPU row-visit test passes, and `setpriv --inh-caps=-all ./tools/check.sh --cuda`
  passes. A Metal row-visit test was added to `lm1b_categorical_fp32.rs` but not yet run. Next on the Mac:
  build and test, check host and device FP32 outputs are identical before/after at batches 1/4/16, and rerun
  `run_lm1b_small_mlx_timing.py`.
- Metal 80-seed campaign and full Metal timing ran at `0a3ce34`; `tools/check.sh --mlx` passed on the Mac
  at `0a3ce34`. The Mac candidate worktree was removed after it was confirmed identical to the commit.
- **Metal sampler change:** a short Metal screen of the old default (host FP64 draws) was 34–47%
  slower than Python MLX, because every level's logits were read to the host and the GPU was
  synced after each level. Metal now defaults to a device FP32 inverse CDF
  (`sample_seeded_device_fp32`, trait hook `lm1b_categorical_8192_fp32`) with non-finite rows
  counted on the device and checked in one read after all levels. `--host-fp64` keeps the old
  sampler. Host FP64 penalties now borrow MLX logits (`visit_f32`). Screen, not evidence (2
  warmups, 5 trials): Rust 0.101/0.185/0.432 s against Python 0.117/0.231/0.658 s at batches
  1/4/16; RSS 1,658/1,705/1,881 MiB against Python ~1,681–1,763 MiB, so batch 16 memory may still fail.
  Checks: four new real-Metal hook tests and the existing top-p tests pass, and Mac Clippy is clean.
  The refactored host path reproduces pre-change Metal outputs exactly at batches 1/4/16, and the
  CUDA rehearsal still reproduces the archive. `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes.
- Metal tooling added, Metal runs pending: the campaign gains a Metal mode that imports the
  published Python CUDA captures by receipt hash and Mac-made Rust captures by pinned binary hash.
  New MLX examples `lm1b_small_cohort_campaign` and `lm1b_small_mlx_timing_probe`, Python probe
  `tools/reference/lm1b_small_mlx_timing_probe.py` (reference forward pass with on-device
  `mx.random.categorical`, as the TinyStories baseline does; the exact reference sampler is a
  pure-Python loop and not a fair speed baseline) and runner `tools/bench/run_lm1b_small_mlx_timing.py`.
  On Linux: synthetic Metal-import self-test (accepts a valid import, rejects a wrong binary hash,
  wrong receipt hash and Metal options in CUDA mode); CUDA-mode rehearsal still reproduces the
  archive exactly; `setpriv --inh-caps=-all ./tools/check.sh` passes. The MLX code is not yet compiled on Apple.
- Prospective 80-seed CUDA campaign passed; the summary replays byte for byte. No code changed.
- The 80-seed campaign tooling is now tracked in `tools/reference/lm1b_campaign/` plus a CUDA example runner.
  A rehearsal with the tracked tool on observed seeds 42/0 reproduces the archived records and comparison
  exactly. Feature Clippy and `setpriv --inh-caps=-all ./tools/check.sh` pass (533 report tests).
  [Evidence](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md).
- The local `.claude/settings.json` permission allowlist is uncommitted and awaits the user's decision.

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

