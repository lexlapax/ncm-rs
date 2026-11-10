# Agent context map

Last updated: 2026-11-10 UTC. Read first; load linked context lazily.

## Current state and user decisions

- Public repository: https://github.com/lexlapax/ncm-rs, `main`, remote `origin`.
- **Phase 2 is complete, including user-requested performance remediation.**
  Final production candidate `083c1e5`; final evidence/acceptance:
  `docs/results/2026-11-10-phase2-acceptance/README.md`. Both platforms pass all
  78 comparisons against previous Rust and qualified Python under unchanged
  limits. Final fresh CUDA, then Metal captures pass 642 traces/2,500 decisions
  each, zero ID disagreements; full real-device checks and tokenizer probes pass.
  CUDA codec latency and Apple checked-load/host-memory deficits are remediated.
  Original/intermediate failures remain historical evidence, not overwritten.
  User authorized implementation, research and commit/push for this extension.
  Phase 3 is not started and needs the next work authorization; no generator/CLI
  work was done. No persistent goal was created or old goal metadata rewritten.
  Large licensed release archives remain separately pending authorization; this
  is a scoped codec/tokenizer result, not full-model inference or a release.
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

## Historical Apple measurement acceptance

- `docs/PLAN.md` and `docs/PHASE2.md` now distinguish the accepted original
  baseline from the reopened performance-remediation extension.
- Independent Python MLX baseline passes all 642 unchanged trained traces and
  2,500 decisions against the CUDA oracle. Three-process codec/tokenizer evidence,
  raw telemetry and baseline qualification are published in
  `docs/results/2026-11-10-phase2-mlx/`. The extension fixes the summary's hidden
  dependency on ignored baseline files using checksum-pinned committed evidence.
  A clean-checkout test reproduces all historical metrics exactly; only current
  summarizer source hashes differ. Historical reports are not rewritten.
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
| Phase 2 contract / acceptance / remediation | [final dual-platform acceptance](docs/results/2026-11-10-phase2-acceptance/README.md), [docs/PHASE2.md](docs/PHASE2.md), [plan](docs/PLAN.md), [research and performance extension](docs/PHASE2-PERFORMANCE.md), [initial research/PoC](docs/results/2026-11-10-phase2-performance-research/README.md) |
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
- Historical native trace: `reports/phase2-codec/cuda-v2`, SHA-256
  `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
  Python reference traces: `reports/phase2-oracle/v1/`. Durable Phase 2 archive
  publication remains pending. Phase 1 archive exists at research prerelease
  `phase1-fixtures-v1`, not an inference release.
- Tokenizers: all four pinned profiles pass 356 encode and 2,314 decode comparisons,
  including bytes/Unicode/specials/postprocessing; Sudoku goldens unchanged.
  Tiny pickle conversion interprets bounded data opcodes, never unpickles or calls
  constructors. Canonical HF JSON comes from offline trusted-code-disabled wrappers.
  Four real-asset mutations/profile are rejected; fresh conversions are identical.
- Historical Linux measurements: `reports/phase2-linux-v1`; all 30 fresh-process raw reports,
  telemetry and first logits are committed under `docs/results/2026-11-10-phase2-linux`.
  Three processes, five warmups, thirty trials; matching output hashes/counts and
  44,856 native benchmark logits pass. B1 reconstruct Rust/Python 18.890/4.668 ms;
  warm codec load 184.49/267.61 ms; host HWM ~559–562/1530–1540 MiB. Native pool
  counters exclude context/cuBLAS allocations; not interchangeable with Torch's
  allocator or device-global snapshots. Occupied workstation/clock variation disclosed.
- Historical instrumented level spans identify intermediate composed resampling as the main
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
- Historical Linux `tools/check.sh --cuda` passes: 30 base Rust tests, real-CUDA codec
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
- **CUDA remediation milestone:** vectorized FP32 gathers/area reduction, pinned
  optional `candle-nn = 0.11.1` fused CUDA norms/softmax, immutable views and explicit
  default-pool retention. No shared model/loader/tokenizer math or goldens changed.
  CUDA default release threshold is now 128 MiB; explicit per-process shared-pool
  API allows zero/custom policy. Not preallocation or a hard cache limit. Tests
  cover real-CUDA policy/opt-out, broad resampling, stable ties and snapshots.
- User authorized stopping `llama-server.service` (inactive, not disabled), closed
  Chrome, then authorized continuing with Hyprland running. No other app was stopped
  or global tooling/power changed. Local CUPTI sample traces identify launch count
  16,756 → 1,312 for load + two B1 reconstructions. B4 bimodality was allocation:
  fast/slow cuMemAllocAsync totals 0.636/20.497 ms with identical kernel work;
  retention gives 0.642 ms. Extra reusable reservation is disclosed (96 vs 64 MiB).
- Predeclared bands are in the research `acceptance-v1.json`, written before
  production changes. Final interleaved old/new/Python reports are archived with
  source/binary hashes, environment and telemetry in the remediation report.
  B1/B2/B4 reconstruction old/new/Python medians: 18.165/3.285/4.659,
  26.681/4.003/5.081, 28.949/5.134/5.859 ms. All codec median/p95 Python bands
  pass descriptively. Warm load 195.88/194.45/286.25 ms; native host HWM ~580 MiB
  remains favorable. Failed vectorized B4 and earlier comparison rounds retained.
- Correctness: final `reports/phase2-codec/cuda-remediated-v1` passes all 642
  trained traces/2,500 decisions with unchanged budgets; same two near ties and
  437/445 input recovery. Native trace SHA-256:
  `d180a52c011c8af15812d10941de8574b1671c2cd23de67e9abc828f7c5ecba4`.
  360 synthetic CUDA rows / 753,984 values pass against pinned Torch CUDA (maximum
  budget fraction 0.628). FP64-coordinate alternative still fails 228 values;
  this GPU check is distinct from the earlier Torch-CPU comparison.
- Validation: final `tools/check.sh --cuda` passed (log
  `.reference/phase2-cuda-remediation-final-check.log`, also published). Final
  `./tools/check.sh` passes, including 20 report tests; 21 offline reference tests
  pass. A clean staged checkout with no ignored inputs exactly reproduces all six
  matrix summaries (historical summarizer provenance excepted), six CUPTI profiles
  and the control summary; evidence/source fingerprints and doc links verified.
  Logs: `.reference/phase2-remediation-publication-check.log` and
  `.reference/phase2-remediation-reference-tests.log`. Staged code/archives reviewed;
  `.pi/`, executables, weights and large model-trace archives are not staged.
- **Extended Linux acceptance passes; next is Apple remediation.** Fixed 30-triplet
  TinyStories and nine-triplet GPT-2 affinity controls close execution uncertainty.
  Qwen's first nine-triplet control still failed uncertainty; a fixed 30-triplet
  identity/path control (only allowed for byte-identical native binaries) passes.
  All negative runs are retained; no tokenizer optimization or proven path cause
  is claimed. Whole-matrix audit passes 78 comparisons with unchanged bands and
  qualified traces. Fresh-process startup/load/execute/drop/exit evidence is now
  separate from steady-state timing. See the Linux acceptance report for metrics.
- Latest `./tools/check.sh --cuda` passes, including 22 report tests; log
  `.reference/phase2-linux-acceptance-check.log`, also published. Audit:
  `reports/phase2-linux-acceptance-v2.json`; controls are archived in the new report.
  A clean staged checkout exactly regenerates acceptance from committed archives
  and passes all 22 report tests; staged scope/diff checked before publication.
  Shared lifecycle harness compiled against both old/current CUDA and exact solved
  outputs agree across all nine processes. No production change in this follow-up.
- Mac availability/state rechecked after `7aee602`: main was clean and safely
  fast-forwarded; old binaries saved in `.reference/phase2-mlx-baseline-bin`, clean
  old source in `.reference/phase2-mlx-baseline` (`5dc9626`). Historical candidate
  worktree inspected and untouched. First remediation diagnostics announced/run.
  Fresh old load ~944 ms and host HWM 4.81 GB reproduce the deficits. Allocation
  stacks identify `bounded_read → RawVec::finish_grow → realloc`; VM maps show
  freed large malloc regions. CPU sample: 2,187/2,222 main-thread samples in SHA-256.
- Shared candidate: reserve checked read length + EOF byte; deserialize authenticated
  BPE/WordPiece tokenizer models directly into the same runtime wrapper; enable
  pinned sha2 0.10.9 `asm` only on macOS/AArch64 (explicit sha2-asm 0.6.4 addition).
  No checksums/preflight/path checks, payloads, precision or algorithms removed.
  Details/evidence: `docs/results/2026-11-10-phase2-shared-loading/README.md`.
- Linux revalidation precedes Apple candidate tests: full `tools/check.sh --cuda`
  passes (`.reference/phase2-shared-load-final-check.log`), all 642 traces/2,500
  decisions and real tokenizer goldens/rejections pass; 21 offline reference tests
  pass. Fresh three-way matrix and 30-triplet controls yield 78 passing comparisons
  (`reports/phase2-linux-shared-load-acceptance-v2.json`). Qwen load 194.80/107.75/
  172.34 ms old/new/Python. Linux codec/tokenizer benchmark binaries remain byte-
  identical after the Apple-only feature addition; Linux SHA features unchanged.
- Shared change pushed as `74c2ed3`, pulled on Mac; full `tools/check.sh --mlx`
  passes. First codec diagnostic: 180.06 ms / 293.63 MB HWM versus old 944.17 ms /
  4.81 GB and Python 270.58 ms / 309.10 MB. Qwen is faster (93.79 ms) but still
  634.52 MB HWM; allocation stack identifies its separate read-buffer reallocations.
- Follow-up reserves checked tokenizer file length + EOF byte, with boundary tests.
  Returned through Linux first: full CUDA checks, real tokenizer goldens/rejections,
  fresh whole matrix and 30-triplet controls pass all 78 comparisons. Negative
  audits retained. Explicit optional `load_controls` applies unchanged paired-process
  uncertainty to checked loads too; malformed-trial regression passes. Evidence
  and exact configs are in the shared-loading report's `raw/` directory.
- SSH restored by user; Mac availability/worktrees rechecked, clean main pulled
  through `2c195ed`. Full Metal checks pass. Tokenizer-buffer candidate improves
  Qwen to 86.97 ms / 455.25 MB HWM, but fresh Python is 145.97 ms / 288.59 MB:
  memory still fails. No unrelated desktop work, tooling or credentials changed.
- User requested reconsidering Rust structures for memory AND performance. New
  candidate packs TinyStories bytes/u32 offsets, stores merge output IDs without
  redundant ranks, borrows reverse lookups and moves valid UTF-8 output storage.
  HF loading avoids full-vocabulary clones for size validation and untagged merge
  trees, keeping the pinned runtime BPE and complete tokenizer visitor unchanged.
  First packed decoder regression retained; ownership fix passes controlled timing.
- Linux first: full CUDA checks, all tokenizer goldens/rejections, complete matrix,
  30-triplet TinyStories/BERT controls and nine full-codec load-control triplets
  pass all 78 comparisons. Qwen load 223.22/89.90/196.78 ms old/new/Python; HWM
  194.49/81.56/240.10 MB. New checked-load control validates every codec report;
  Apple controls explicitly have no affinity claim and require activity telemetry.
  New MLX lifecycle/three-way tools await real Apple execution. Full details and
  all failed/final evidence: `docs/results/2026-11-10-phase2-compact-tokenizers/`.
- Compact candidate pushed/pulled as `45743d0`; full Metal checks and real
  tokenizer goldens pass. Qwen improves to 65.82 ms / 326.83 MB HWM but still
  exceeds Python's memory band. Retained as failed memory diagnostic. Follow-up
  reserves canonical BPE vocab/merge containers from pinned profile capacities;
  tests enforce bounds. Full Linux CUDA checks and fresh matrix + 30-triplet
  controls again pass all 78 comparisons; unchanged-codec load control is explicitly
  reused with source binding. Evidence appended to the compact-tokenizers report.
- Capacity follow-up pushed/pulled as `083c1e5`. Full Metal checks and real
  tokenizer goldens pass; 642 trained traces/2,500 decisions qualify unchanged,
  zero ID disagreement. Fresh Apple matrix: codec load 945.87/179.99/272.21 ms
  old/new/Python, HWM 4749.16/293.91/308.69 MB; Qwen 189.29/63.65/144.78 ms,
  HWM 633.88/69.81/290.98 MB. Repeated-load probes plateau with zero active MLX
  bytes after drops. All latency point estimates and host-memory ratios pass;
  75/78 cells meet conservative process envelopes. GPT-2 encode and Qwen
  encode/decode versus old Rust still need fixed additional process controls.
- Separate `.reference/phase2-mlx-lifecycle-baseline` at `5dc9626` has ONLY copied
  lifecycle example/support files; baseline build and remote lifecycle command
  succeeded. **SSH agent became unavailable again before report transfer.** Retry
  denied; user asked to unlock. Mac lifecycle report/final check logs remain remote;
  do not infer lifecycle figures or final acceptance. Next use a process-local
  persistent SSH connection after access returns, without global configuration edits.
- Published qualified Apple measurements and explicit pending diagnostic:
  `docs/results/2026-11-10-phase2-mlx-remediation/README.md`. Audit now binds Python
  control sources as well as native ones; mutation regression and local checks pass,
  latest Linux gate still passes 78 comparisons under the stronger audit. A clean
  staged checkout without ignored artifacts exactly reproduces Linux acceptance
  and both Apple summaries; all 24 report-tool tests pass.
- Final follow-up: SSH restored; availability rechecked and a process-local
  persistent connection used (no global SSH edits). Mac main pulled `42caaec`.
  Retrieved/validated lifecycle and final logs. Fixed 30-triplet GPT-2/Qwen controls
  resolve all three uncertain cells: all 78 Apple comparisons pass. Linux also
  passes 78 under the final audit. All negative rounds remain archived.
- Final `tools/check.sh --cuda`, then `--mlx`, pass; fresh native trained traces
  captured CUDA first then Metal both pass 642/2,500 unchanged. Cleared-environment
  tokenizer probes pass on both. Evidence/configs/logs are in the final report.
  Apple lifecycle medians: old/new/Python full process 1050.72/287.30/468.01 ms;
  startup 39.84/39.68/106.10 ms. Not disk-cold or steady-state timing. Clean staged
  checkout exactly reproduces both 78-cell audits and all four matrix summaries;
  all 24 report tests and evidence hashes pass. Control binaries match matrices.
  All 21 offline reference-tool tests and the final default publication check pass.
  Plan, tracker, extension contract, root README, testing/architecture status and
  results index now consistently record completion without a generator/release claim.
- **Next: await the next task; Phase 3 is unstarted, not blocked by Phase 2.**
  Preserve historical worktrees, goldens and `.pi/`. No service, unrelated process,
  clock/power policy or global tooling was changed for the final measurements.
