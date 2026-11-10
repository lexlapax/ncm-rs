# Agent context map

Last updated: 2026-11-10 UTC. Read first; load linked context lazily.

## Current state and user decisions

- Public repository: https://github.com/lexlapax/ncm-rs, `main`, remote `origin`.
- **Confirmed goal: complete Phase 2. Linux/CUDA and Apple/MLX gates are complete**,
  including codec/tokenizer parity and matched three-process performance/memory
  evidence. Linux native codec is slower than eager Torch; Apple native host RSS
  is 3.40–4.42 GiB versus the adapted Python MLX baseline's 0.287–0.288 GiB.
  Neither platform result is an efficiency, generation or full text-model claim.
  SSH authentication was restored by the user; the first Phase 2 MLX run followed
  Linux acceptance. Acceptance evidence commit `7a74fb3` is pushed; durable
  large-archive release publication remains separately pending authorization.
- Within EVERY phase: finish Linux/CUDA work and validation, then macOS/MLX.
  Both gates before advancing. Phase 3 generator/CLI work is out of this goal.
- Phase 1 is complete on both platforms ([acceptance](docs/PHASE1.md)). Phase 2
  implements shared FP32 Sudoku codec semantics and TinyStories/GPT-2/Qwen/BERT
  text tokenizers, **not full text-model inference**. No generator, inference CLI,
  official task-quality result, generalized importer or inference release exists.
- Production execution stays Python-free. Offline conversion/reference/measurement
  and native CUDA/MLX libraries are expected. Never silently change pins, schemas,
  established goldens or numerical tolerances; unsupported options must fail.
- Local checks replace GitHub build CI: `tools/check.sh`, opt-in `--cuda`/`--mlx`.
  Never restore build CI without approval. Metadata-only upstream watcher remains.
- `ssh macstudio` is authorized. Recheck availability/competing work before later
  runs, inspect worktree, then pull reviewed pushed commits into
  `~/projects/lexlapax/ncm-rs`. Never overwrite unrelated work or global tooling.
  Announce the first Phase 2 MLX run only after Linux acceptance.
- Update this map before EVERY commit, including amend/fixup. No agent attribution
  or co-author trailers; preserve upstream licenses and technical provenance.
- Persistent goal state is untracked `.pi/`; leave it untouched and unstaged.

## Apple measurement acceptance

- Independent Python MLX baseline passes all 642 unchanged trained traces and
  2,500 decisions against the CUDA oracle. Three-process codec/tokenizer evidence,
  raw telemetry and baseline qualification are published in
  `docs/results/2026-11-10-phase2-mlx/`; its fail-closed summary reproduces
  byte-for-byte from committed raw reports.
- Host-only artifact loads reproduce high native RSS; diagnostics remain at
  `reports/phase2-mlx-load-diagnostic`. No loader fix or efficiency claim.
- Final summary-tool tests pass locally. On the M1 Ultra candidate worktree,
  `./tools/check.sh --mlx` passed on 2026-11-10 with the pinned MLX prefix,
  including real-Metal components/operators and 11 report-tool tests; the exact
  measured source hashes were verified equal to the candidate files before this
  check. MLX reference asset tests also pass in `.reference/mlx-venv`.
- Preserve `.pi/` and the Mac candidate worktree's unrelated/uncommitted state.

## Task routing

| Need | Load |
| --- | --- |
| Phase 2 contract / acceptance | [docs/PHASE2.md](docs/PHASE2.md), [plan](docs/PLAN.md) |
| Shared runtime and model | [core](crates/ncm-core/src/lib.rs), [config](crates/ncm-core/src/config.rs), [model](crates/ncm-model/src/lib.rs), [blocks](crates/ncm-model/src/blocks.rs), [codec](crates/ncm-model/src/codec.rs) |
| CUDA adapter / known native fixes | [adapter](crates/ncm-backend-cuda/src/codec.rs), [operators](crates/ncm-backend-cuda/src/ops.rs), [component report](docs/results/2026-11-10-phase2-primitives/README.md) |
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

## Evidence and boundaries

- Phase 1: all 18 fixed operator cases pass CPU/CUDA/MLX at unchanged
  `atol=1e-5, rtol=1e-4`; all 238 checkpoint tensors read back byte-exact.
  Metadata-only manifest integrity is NOT authenticity or semantic validation.
  Root/leaf symlinks are rejected, but trusted parents/no concurrent hostile
  mutation remain required. Input byte limits are NOT peak-RSS bounds.
- Historical Linux checked-load median Rust/Python ~202/241 ms. Apple ~939/282 ms
  with repeated-load host peak RSS ~4.40/0.41 GiB: Rust slower/higher RSS. Native MLX
  active arrays return to zero, cache retains ~155 MiB; counters do not explain
  total host RSS. Investigate in Phase 2 before any efficiency claim.
- Phase 1 composed resampling was slower than PyTorch. Compiled Torch FP16 loses
  rounding on one fixture: compiled mixed precision remains unsupported, no waiver.
- Phase 2 reference budgets were declared before implementation in `docs/PHASE2.md`.
  Canonical trained oracle is CUDA; tiny oracle is CPU. Original-Python trained
  CPU/CUDA diagnostic retains one failing blank-board level-zero projection value
  (22,784-term dot, channel 113, ~0.001054 difference). Never use this to exempt
  candidate GPU failures or silently relax tolerances.
- Config/loading: strict initial codec family, exact 58 F32 runtime tensors,
  checked IDs/options, full host preflight before device upload; training buffers
  are separate. Both tiny and trained inventories match. No unsafe Send/Sync.
- Codec `n_layers=4` means head count, not decoder depth. Decoder is bidirectional,
  Q/K RMS-normalized and exact-erf GELU. GroupNorm spans positions/grouped channels.
  Preserve exact area bins, half-pixel interpolation and independent snapshots.
- Candle fixes: CPU batched transposed convolution misindexes a multi-position
  second batch, so only CPU dispatches separate batch slices. CUDA remains native
  batched. Native CUDA argmin tie order was wrong; selection now reduces values,
  then masked indices on-device. Tests cover wide/signed-zero ties and both fixes.
- Complete tiny codec: all 236 traces pass CPU/CUDA; snapshot/uninstrumented checks
  yield 252 floating comparisons / 112,212 elements; all IDs exact.
- Trained CUDA codec: all 642 traces and 2,500 code/output decisions pass. Two
  mixed-board near ties (levels 1/2) remain visible. Worst normalized error uses
  **94.2%** of the budget at blank-board level-zero channel 113. Do not infer Apple
  parity from Linux's narrow margin. Two fresh native traces are byte-identical.
- Reconstruction matches Python on 445/445 tokens and 5/5 sequences; recovery of
  original inputs is 437/445 (eight mixed-board differences), not losslessness or
  Sudoku-solving quality. Uninstrumented outputs match traced outputs.
- Current native trace: `reports/phase2-codec/cuda-v2`, SHA-256
  `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
  Python reference traces: `reports/phase2-oracle/v1/`. Durable Phase 2 archive
  publication remains pending. Phase 1 archive exists at research prerelease
  `phase1-fixtures-v1`, not an inference release.
- Tokenizers: all four pinned profiles pass 356 encode and 2,314 decode comparisons,
  including bytes/Unicode/specials/postprocessing; Sudoku goldens unchanged.
  Tiny pickle conversion interprets bounded data opcodes, never unpickles or calls
  constructors. Canonical HF JSON comes from offline trusted-code-disabled wrappers.
  Four real-asset mutations/profile are rejected; fresh conversions are identical.
- Linux measurements: `reports/phase2-linux-v1`; all 30 fresh-process raw reports,
  telemetry and first logits are committed under `docs/results/2026-11-10-phase2-linux`.
  Three processes, five warmups, thirty trials; matching output hashes/counts and
  44,856 native benchmark logits pass. B1 reconstruct Rust/Python 18.890/4.668 ms;
  warm codec load 184.49/267.61 ms; host HWM ~559–562/1530–1540 MiB. Native pool
  counters exclude context/cuBLAS allocations; not interchangeable with Torch's
  allocator or device-global snapshots. Occupied workstation/clock variation disclosed.
- Instrumented level spans identify intermediate composed resampling as the main
  native cost. Original-Python fullgraph Inductor preflight fails on NumPy-based
  Phi module indexing; no compiled timing baseline is qualified. Tokenizer timings
  retain both skip modes and special/non-special throughput; Qwen native loading
  is slightly slower in pooled medians. Original-source vs canonical load formats
  differ. No model/operator/tokenizer math, dependency pin or golden changed.
- MLX now implements the existing backend contract with opaque immutable arrays
  and an explicit GPU stream (restores a CPU-scoped caller); no model duplication.
  All 18 operators, 140 components and complete tiny codec pass; 642 trained traces
  and all 2,500 decisions pass. Two fresh captures are identical, SHA-256
  `f20ffe0f36a82c80a81e91a61b61950968821369b6b41d0f13cbf9e732875a0f`.
  Same two near ties; reconstruction 445/445 versus Python, input recovery 437/445.
- Initial MLX full-span convolution failed two blank-board projection values despite
  exact IDs. Retained in the Apple report. Equivalent native FP32 matmul for
  zero-padding/full-span convolution fixes parity; no precision/oracle/budget change.
  Worst component still uses 94.28% of budget. Large v1/v2 traces remain ignored
  under `reports/phase2-codec/` on both machines; Mac also has `mlx-v2-repeat`.
- The shared trained-capture harness refactor preserves the CUDA trace hash exactly.
  Mac tokenizer probe passes all four profiles, including rejection cases and a
  cleared environment without Python. Canonical tokenizer assets are now on Mac.
- Latest Linux `tools/check.sh --cuda` passes: 30 base Rust tests, real-CUDA codec
  suite, all 18 unchanged operators, fmt/Clippy/docs/syntax. Twelve report-tool tests
  and all 18 offline reference tests pass. Tokenizer probe previously also passed
  with cleared env and no Python on PATH. Logs: `.reference/phase2-linux-check.log`,
  `.reference/phase2-linux-reference-tests.log`; reports retain exact fingerprints.
  Shared-harness/final Linux checks: `.reference/phase2-mlx-shared-linux.log`,
  `.reference/phase2-mlx-final-linux.log`; all 18 offline tests also rerun.
  Mac `tools/check.sh --mlx` passes with four new codec tests; logs copied locally:
  `.reference/phase2-mlx-v2-check.log`, `.reference/phase2-mlx-v2-components.log`.

## Provenance and next actions

- Phase 1 closure `650cc7d`; Phase 2 oracle `97a3125`, checked runtime `8492c27`,
  components `a4a6344`, complete codec `bb77fef`, tokenizers `0396b56`, Linux runtime
  gate `fd1b482` (pushed). Preserve old reports/source hashes;
  do not relabel historical runs. Benchmark outputs are synthetic fixtures, not
  task-quality data or an inference release.
- **Next: Phase 2 is closed.** Publish large licensed oracle/native/tokenizer
  archives as durable release assets only if authorized; their absence remains
  visible. Do not begin Phase 3 or claim an inference release without a new goal.
