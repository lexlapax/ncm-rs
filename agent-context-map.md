# Agent context map

Last updated: 2026-11-10 UTC. Read first; load linked context lazily.

## Current state and user decisions

- Public repository: https://github.com/lexlapax/ncm-rs, `main`, remote `origin`.
- **Current confirmed goal: complete Phase 2. In progress on Linux; neither Phase 2
  runtime gate is passed.** Reference contract and checked shared-runtime construction
  and complete Linux codec trace parity are implemented. Tokenizer assets and
  Linux performance/memory acceptance are next; Apple codec work has not started. See [Phase 2 contract](docs/PHASE2.md).
  Phase 1 passed Linux/CUDA then macOS/MLX ([acceptance](docs/PHASE1.md)). Keep this
  same order within EVERY phase; both gates before advancing. Phase 3 is out of scope.
- The confirmed Phase 2 goal includes TinyStories byte-BPE and GPT-2/Qwen/BERT
  tokenizer asset conversion/parity, not full text-model inference. Goal state is
  managed by the harness under untracked `.pi/`; do not edit or stage that state.
- Implemented: candidate CPU/CUDA/MLX operators, shared fixed fixture contract,
  scoped Sudoku exporter/checked copied loader, Sudoku text tokenizer, upstream
  model traces and reproducible operator/load/tokenizer/memory reports.
- Shared FP32 codec now passes tiny CPU/CUDA and trained CUDA traces. **No Rust
  generator, inference CLI, official task-quality result, generalized importer or
  inference release.** Codec output parity is not the complete Phase 2 platform gate.
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
| Current Phase 2 contract / fixture audit | [docs/PHASE2.md](docs/PHASE2.md), [oracle audit](docs/results/2026-11-10-phase2-oracle/README.md) |
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
  cached generation, packaging and training remain pending (tokenizer assets are
  current Phase 2 work). OWT weights and
  the upstream full training pipeline remain unavailable.

## Phase 2 reference-contract milestone

- Audited all pinned codec/quantizer source and the 58 runtime tensor names/shapes.
  Important: codec `n_layers=4` is head count; decoder is bidirectional, unlike the
  common generator attention's default causal path. Exact-erf GELU is required.
- Declared `codec-fp32-v1` budgets before Rust codec work in `docs/PHASE2.md`.
  Existing operator tolerances/goldens are unchanged. Full-checkpoint CUDA is the
  canonical trained oracle; tiny synthetic CPU is the portable component oracle.
- Added `tools/reference/codec_trace.py`, comparator and four utility tests. Captures
  clone every snapshot; decode shadow sums match the original decoder latent, and
  encode/decode snapshots agree. No upstream source/math patches or pickle loading.
- New `tests/fixtures/codec-v1.json`: 170 interned typed arrays (~608 KiB), untrained
  width-16/length-16 codec, two-stage normalized cascade, 236 traces across two cases.
  Full Rust execution now passes CPU/CUDA (see complete-codec milestone below).
- Trained Sudoku CUDA: 642 traces, solved B2 / blank B1 / mixed B2, with free encode
  and fixed-code decode. Two fresh canonical generations have identical file hashes.
  Phase 1 solved-batch codec input/pre-quant/codes/logits remain byte-exact.
- Tiny original-Python CPU/CUDA comparison passes all 236 traces. Real-checkpoint
  CPU/CUDA diagnostic retains one failing internal component on blank-board level-0
  learned downsampling (22,784-term projection; channel 113 differs by ~0.001054).
  All IDs and final reconstruction/logit budgets agree. No tolerance widened; this
  diagnostic cannot exempt a future GPU candidate from the trained-CUDA oracle gate.
- Nine optional reference tests pass; every fixture array's key/shape/dtype/byte
  count and reference was checked. Small reports are public; large traces (~92 MiB
  for trained capture) are ignored in `reports/phase2-oracle/v1/`; durable Phase 2
  archive publication is still pending. This historical milestone was reference-only.
- Reconciled stale README/plan/testing status. Runtime code is unchanged in this
  milestone. Full `tools/check.sh --cuda` passes (15 Rust tests, seven report-tool
  tests, all 18 real-GPU operator cases, lint/fmt/docs/syntax), plus nine optional
  reference tests. Log: `.reference/phase2-contract-check.log`. Hashes/links pass.

## Phase 2 shared-runtime milestone

- Added strict typed/validated codec config and exact runtime weight inventories
  in `ncm-core`; `n_layers` remains explicitly head count. Both tiny cascade and
  trained inventory names/shapes match their committed references. Unknown codec
  options, unsupported architecture, invalid probabilities/IDs and bounds fail.
- Added `WeightBackend` / provisional `CodecBackend`, immutable-array semantics,
  no Send/Sync bounds, FP32 BLC/BHLD layouts, first-index ties, exact GELU and
  explicit bidirectional attention. No duplicate backend-specific model logic.
- `ncm-model::Codec` preflights all names/shapes/dtypes/finiteness before allocation;
  copied endian decoding retains only one temporary tensor. Loader byte bounds
  are not peak-RSS bounds. Artifact authenticity remains caller responsibility.
- Seven new Rust tests pass (22 total), including rejection before any allocation,
  both inventories, unsupported configs, code/token ranges and allocation errors.
  Full `tools/check.sh --cuda` passes, including unchanged 18 GPU operator cases;
  log `.reference/phase2-runtime-check.log`. No Phase 2 Apple run.
- `checked_codec_load` release example constructs the shared codec on CPU and real
  CUDA: all 58 runtime tensors / 59,586,048 bytes read back byte-exact. Reproduction
  and raw summaries: [runtime checks](docs/results/2026-11-10-phase2-runtime/README.md).
  No forward execution/performance qualification implied; load/RSS and affected
  workload benchmarks remain pending the Linux gate, then Apple.

## Phase 2 Linux primitive/component milestone

- Implemented `CodecBackend` for Candle, shared components in
  `crates/ncm-model/src/blocks.rs`, and device-neutral tiny fixture runner at
  `tests/support/codec.rs`. Includes complete traced decoder block, exact GELU,
  normalized cascades, Euclidean codebooks, Phi refinement and accumulation.
- CPU/CUDA each pass 140 floating comparisons / 75,348 values; largest absolute
  errors ~1.1444e-5 / 7.6294e-6. All selected tiny code IDs exact. Component tests
  use reference residual inputs; complete free-running encoding is now separately
  qualified below.
- Two regression fixes without pin/tolerance changes: Candle CPU batched ConvT
  misindexes a multi-position second batch, so CPU dispatches separate batch
  slices; CUDA remains batched. CUDA native argmin picks a non-first tied index,
  so adapter selection uses value reduction plus masked-index minimum on-device.
  Tests cover both fixes, signed-zero/wide-row ties and retained snapshots.
- `tools/check.sh --cuda` now runs native CUDA release tests in addition to the
  old probe. Full checks pass: 24 base Rust tests plus real-CUDA component suite,
  seven report tests, all 18 unchanged GPU operators, lint/fmt/docs/syntax. Log:
  `.reference/phase2-primitives-check.log`. No Phase 2 Apple run.
- [Component report](docs/results/2026-11-10-phase2-primitives/README.md) retains raw
  release output and defects/workarounds. No performance or trained-model claim;
  complete codec traces are now separately qualified below; measurements remain pending.

## Phase 2 complete shared codec milestone

- `crates/ncm-model/src/codec.rs` implements encoder/pre-quant, residual quantization,
  decoder, encode/decode and reconstruction with on-device codes and retained
  independent snapshots. Instrumented and uninstrumented outputs agree.
- All 236 tiny traces pass CPU/CUDA; extra snapshot/uninstrumented checks yield
  252 floating comparisons / 112,212 elements. All tiny IDs exact.
- Native `codec_trace` authenticates reference inputs/config/weights; all 642 trained
  CUDA traces pass. All 2,500 code/output decisions agree; two mixed-board near ties
  (levels 1/2) remain visible. Two native processes produce identical trace hashes.
- Worst normalized error is blank-board level-zero channel 113, using 94.2% of its
  unchanged budget. No tolerance or golden changes. Original-Python CPU diagnostic
  failure remains separate; do not infer Apple parity from Linux's narrow margin.
- Reconstruction agrees with Python on all 445 tokens / five sequences. Original
  input recovery is 437/445 tokens (eight mixed-board differences), not losslessness.
- [Complete codec report](docs/results/2026-11-10-phase2-codec/README.md) has source
  hashes, per-tensor errors, histograms, stable decisions and reconstruction counts.
  Large native traces: `reports/phase2-codec/cuda-v2`; SHA-256
  `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
- Full `tools/check.sh --cuda` passes (24 base Rust tests plus real-CUDA suite,
  unchanged 18 operators, seven report tests, lint/fmt/docs/syntax), plus 13 offline
  reference tests. Log `.reference/phase2-codec-check.log`. New qualifier checks
  candidate selection consistency, stable margins and all float budgets separately.
- No latency/throughput/memory claim yet. Durable Phase 2 trace archive remains
  pending publication. No Phase 2 Apple work; Linux acceptance must precede it.

## Latest provenance and next actions

- CUDA milestone `3178aa1`; local checks `f7aad49`; artifacts/traces `bbdf422`;
  Linux load gate `e8b81c4`; MLX candidates `ee5f51a` / `8f18246`; measured operator/
  Rust load sources `2d262e1`; matched Python load preflight/aggregation `2b19c87`.
- Final Apple raw trials: `reports/phase1-final/mlx` and `mlx-load-matched` locally;
  remote `reports/phase1-mlx/final` and `load-final`. Final remote local-check log:
  `reports/phase1-mlx/check-closure.log`; Linux `.reference/phase1-linux-closure.log`.
- Phase 1 closure is pushed as `650cc7d`; Phase 2 reference contract as `97a3125`,
  checked shared runtime as `8492c27`, components as `a4a6344`.
  Reference reports embed exact helper hashes.
- Next: safe pinned TinyStories/HF tokenizer assets and exact parity; then matched
  Python/native load, latency, throughput and memory measurements. Only then start Phase 2 Apple work. Preserve the fixed oracle
  budgets, retain near-margin cases and negative findings, and do not conflate
  reference generation with Rust codec qualification.
