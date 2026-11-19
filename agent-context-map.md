# Agent context map

Last updated: 2026-11-19 UTC. Read first; load linked context lazily. Keep this file short:
current state, decisions, blockers and next actions. Put durable evidence in `docs/results/`.
The full pre-compaction history is archived in [docs/history/context-map-2026-11-18.md](docs/history/context-map-2026-11-18.md).

## Current state

- **Phases 0–3 complete.** Phase 2 closed with a [scoped Apple immediate-memory exception](docs/results/2026-11-11-phase2-optimization/phase2-final-audit-v1.md).
  Phase 3 Sudoku is complete on CUDA and Metal ([report](docs/results/2026-11-14-phase3-sudoku/README.md)).
- **Phase 4 complete with scoped, user-accepted exceptions** (2026-11-19, [audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md), [contract](docs/PHASE4-TEXT.md), [handoff](RESUME.md)).
- **Phase 5 step 1 in progress** ([PHASE5.md](docs/PHASE5.md)): user chose release first, lean, and FP32 only (2026-11-19). `ncm sample` routes supported profiles to existing paths; ordinary GPU CLI features enable the prior experimental feature paths. A [tracked CUDA routing runner](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) matches baseline, alias and unified stdout for one request per profile, including all three Sudoku difficulties; a [decode screen](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md) matches the pinned references on those text outputs. XSum/WMT14 `--text` has [corpus-exact Linux encoding and CUDA end-to-end screens](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md); [LM1B's default rotary sidecar](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md) preserves CUDA output without a command-line rope argument; Metal validation remains pending. TinyStories `--text` now accepts strings encoding to exactly one valid non-padding token, with a [CUDA screen](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-text-input); LM1B `--text` behavior awaits the user's decision. [Linux export investigation](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md) reproduced seven qualified manifests exactly; Sudoku medium/hard require a missing historical exporter or an explicit provenance-only re-export decision. [Setup guide](docs/USAGE.md) is incomplete; a [fresh-clone Linux Sudoku-easy slice](docs/results/2026-11-19-phase5-cli/linux-fresh-clone-sudoku-v1.md) passes, but complete fresh-clone verification, performance and the release gate remain pending.

| Profile | Status |
| --- | --- |
| TinyStories | Advanced under [user-accepted exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md) |
| XSum | Advanced under the [Metal](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md) and [CUDA stochastic](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md) exceptions |
| WMT14 | Scoped CUDA and Metal top-1 qualification ([evidence](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification)) |
| LM1B-small | **Qualified on CUDA and Metal:** 80-seed quality passes on both; timing passes on both, including Metal RSS after the [memory fix](docs/results/2026-11-16-phase4-lm1b/small-metal-timing-v1.md#after-the-memory-fix-all-checks-pass) |
| LM1B-medium/large | **Lean bar met on CUDA and Metal:** teacher-forced parity ([CUDA](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-parity-v1.md), [Metal](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md)), 80-seed quality ([CUDA](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-results-medium-and-large), [Metal](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-metal-results-medium-and-large)), CLI generation, and timing faster than Python ([CUDA](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md), [Metal](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md#latency-and-memory-against-python-mlx)) |
| OWT | Config-only coverage done: pinned config validated (`OwtConfigOnly`), artifacts rejected, synthetic operators at OWT shapes match naive references ([coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md)). Weights unreleased |

- **LM1B-small evidence:** CUDA fixed-history traces, greedy and CLI checks pass. Repeated CUDA
  timing beats pinned Python in median, p95 and process RSS at batches 1, 4 and 16; Rust model load
  is slower ([performance](docs/results/2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements)).
  The frozen five-seed entropy interval failed on CUDA and every observed-seed retest; the
  failures stay on record ([quality](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md)).
  The approved **80-seed CUDA gate passes**: perplexity +0.44 [-0.27, 1.15], entropy +0.00073
  [-0.00032, 0.00180] ([result](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-result)).
  Metal fixed-history decisions and IDs match CUDA; draw-level differences are covered by
  user decision 2 below.
  The independent [Python MLX reference](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#metal-development-reference-for-complete-requests)
  is `tools/reference/lm1b_small_mlx_reference.py`.

- **Metal results (small):** the 80-seed gate passes, with perplexity +0.29 [-0.41, 0.97] and entropy +0.00103
  [-0.00007, 0.00213] ([result](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-metal-result)).
  After the memory fix (`57e5ee6`, output byte-identical), timing passes median, p95 and RSS at batches 1/4/16
  (13/15/28% faster; RSS margin 0.4–0.6%) ([timing](docs/results/2026-11-16-phase4-lm1b/small-metal-timing-v1.md)).
- **macstudio is always available** for testing; recheck its state before use.

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

Run checks under `setpriv --inh-caps=-all`; the shell may carry an inheritable capability.

1. **Phase 5 step 1 (CUDA, then Metal):** complete unified `ncm sample` (LM1B defaults to a separately exported, hash-pinned sibling rotary file), confirm byte-identical output against the current commands; finish validating the `*-experimental` feature folding; validate `--decode` and finish `--text`
   for every text profile; XSum/WMT14 corpus encoding passes on Linux, CUDA end-to-end is screened, Metal remains pending; `docs/USAGE.md` verified from a fresh clone; README and release
   notes; then ask before tagging `v0.1.0`.
2. Steps 2–4 follow [PHASE5.md](docs/PHASE5.md): profile, user picks targets, targeted optimizations, close.
3. **Retained local inputs** after the 2026-11-19 cleanup: pinned upstream checkpoints and evaluators, one current
   artifact per profile, the reference environments, `reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors`
   and `reports/phase4-wmt14-uncached-cfg-link-v1/` (Linux), and `reports/phase4-lm1b-metal-cli-rope-v1.safetensors` (Mac).
   Raw Phase 4 captures were deleted; the docs keep their hashes.

## Working rules carried forward

- Linux/CUDA first, then Metal, for each step. `ssh macstudio` is authorized; inspect the clean
  remote checkout and fast-forward pull pushed commits; do not reboot the Mac or interrupt `dotsetup.rmlx`.
- Keep `.pi/` untouched. Ignored `reports/` hold raw captures; tracked docs hold identities and conclusions.
- CUDA check: `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.

## Checks run for this update

- **Unified CLI rejection integration:** [ten synthetic-manifest cases](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md#cli-rejection-integration-checks) exercise actual CLI rejection diagnostics and no stdout for unsupported profile/option combinations and two LM1B device/flag cases. Default and `cuda-cudnn` test builds, CUDA CLI all-target Clippy and `tools/check.sh` pass. All cases stop before artifact loading or inference; this is not a complete device matrix or GPU validation.

- **Integrated fresh-clone Linux guide replay:** the CUDA build/check, tokenizer setup, seven fresh downloads/qualified exports, both rotary references and synthetic Sudoku/text examples passed in one new clone; its inspected worktrees and clone were removed ([evidence](docs/results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md)). `tools/check.sh --cuda` reported 18 passing operator results; 15 guide shell blocks and links pass syntax/link checks. Sudoku medium/hard, full Mac guide, broader parity and quality/performance gates remain pending.

- **Synthetic text examples:** added [draft CUDA commands](docs/USAGE.md#synthetic-text-examples--cuda-development-screen) for TinyStories, XSum, WMT14 and LM1B-small. Four one-row requests passed with retained qualified artifacts and decoded text ([scope](docs/results/2026-11-19-phase5-cli/cuda-synthetic-text-examples-v1.md)); 14 guide shell blocks, links and `tools/check.sh` pass. They are smoke tests, not fresh-clone, Metal, parity, quality or performance evidence.

- **Fresh-clone Linux Sudoku guide replay:** extracted and followed the guide's CUDA build, `tools/check.sh --cuda` (18 operator results), checkpoint download, historical export and synthetic `sample`/`generate` blocks; the qualified manifest and identical output passed. The inspected clone was removed ([evidence](docs/results/2026-11-19-phase5-cli/linux-guide-sudoku-replay-v1.md)); 13 guide shell blocks and links pass syntax/link checks. Only one profile was replayed, not the full Linux or Mac guide, quality or performance gate.

- **Fresh-clone Linux tokenizer slice:** followed the guide's upstream clone, tokenizer venv, pinned network fetch and conversion from a new clone; all four tokenizer manifests and payloads matched the tracked lock, and the inspected temporary clone was removed ([evidence](docs/results/2026-11-19-phase5-cli/linux-fresh-clone-tokenizers-v1.md)). Added the exercised hash check to the guide; thirteen shell blocks, links and `tools/check.sh` pass. Full setup replay, model execution and Metal remain pending.

- **LM1B sidecar output safety:** offline reference exporter now exclusively creates its output, rejecting existing files and live/dangling symlinks, with three dependency-free tests added to `tools/check.sh`. A fresh Linux reference export exactly matches the pinned sidecar and the prior bytes ([evidence](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md)); `tools/check.sh` passes. This is not new GPU, quality or fresh-clone evidence.

- **WMT14 and LM1B export recipes:** added [Linux draft commands](docs/USAGE.md#linux-wmt14-and-lm1b-development-replays) for WMT14, all LM1B sizes and the shared rotary sidecar. Retained inputs reproduced the four qualified manifest hashes and sidecar hash ([scope](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#wmt14-and-lm1b-guide-command-replays)); twelve shell blocks, document links and `tools/check.sh` pass. No fresh-clone, Mac, inference or quality gate is implied; complete guide replay and Sudoku medium/hard decision remain pending.

- **XSum export recipe:** added the pinned rotary preparation and historical exporter to the [draft guide](docs/USAGE.md#linux-xsum-development-replay); retained inputs reproduced both the rotary payload and qualified manifest ([scope](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#xsum-guide-command-replay)). Nine guide shell blocks, two rotary tests and `tools/check.sh` pass. No fresh-clone, Mac, inference or quality gate is implied; remaining exporter recipes and complete guide replay are pending.

- **TinyStories export recipe:** added a historical exporter and qualified manifest check to the [draft guide](docs/USAGE.md#linux-tinystories-development-replay); retained sources verified and the export replay matched the qualified manifest ([scope](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#tinystories-guide-command-replay)). This did not use a fresh clone or download; the full guide and remaining exporters are still pending. Eight guide shell blocks pass `bash -n`; `tools/check.sh` initially failed the intermittent clock-handoff test, then its four focused tests and a full retry passed (cause unknown).

- **CUDA CLI cold-request screen:** [Sudoku-easy report](docs/results/2026-11-19-phase5-cli/cuda-sudoku-cold-screen-v1.md) compares baseline vs unified CLI over 90 alternating fresh processes per condition (plus warmups), with byte-identical outputs. Medians 555.115/554.145 ms and median peak RSS 587.965/587.773 MiB; no meaningful improvement established. Earlier timed-wait screens had biased latency and are invalid. This includes load/process startup, not warm in-process inference, and lacks native device peaks, Python comparison, other profiles and Metal. Five runner tests and default `tools/check.sh` pass. The Phase 5 performance gate remains pending.

- **TinyStories vocabulary text differential:** pinned upstream and Rust agree on acceptance and admitted ID for 8,199 decoded-vocabulary/boundary texts (8,064 accepted, 135 rejected); rejected full ID sequences are not compared. [Reference generator and reproduction](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-vocabulary-differential), ignored artifact-bound Rust test and two generator-preflight tests pass. CLI feature tests/Clippy and two final `tools/check.sh --cuda` runs pass; an earlier unrelated clock-handoff observation mismatch failed and is recorded. No new GPU quality/performance evidence is implied.

- **Fresh-clone Linux Sudoku-easy slice:** a new remote clone installed the Linux reference lock, built CUDA CLI, downloaded pinned sources, reconstructed the historical exporter and qualified manifest, compared synthetic sample/alias output, and passed `tools/check.sh --cuda` (18 GPU cases). The clone and inspected worktree were removed. [Evidence](docs/results/2026-11-19-phase5-cli/linux-fresh-clone-sudoku-v1.md); other profiles, documentation replay and Metal remain pending. Corrected the earlier easy-exporter SHA transcription, checked documentation links and shell syntax, and ran default `tools/check.sh`.

- **Bounded checkpoint downloads:** added [fetch_checkpoints.py](tools/reference/fetch_checkpoints.py), reusing existing exporter pins with explicit byte budget, streamed size/hash checks, plan and verify modes. All nine retained profile sources verify; a fresh 162,378,392-byte Sudoku-easy network download matches retained files, and its export's non-provenance bytes match the qualified artifact ([screen](docs/results/2026-11-19-phase5-cli/checkpoint-download-screen-v1.md)). Seven downloader tests and final `tools/check.sh` pass. The first full check failed an unrelated lifecycle `ObservationMismatch`; a focused retry and two full reruns passed, cause unknown. Full fresh-clone workflow remains pending.

- **Setup guide draft:** [docs/USAGE.md](docs/USAGE.md) documents current toolchain, pinned-source and download layout, with incomplete export and fresh-clone steps explicitly marked. Eleven small checkpoint-source files (nine configs and two TinyStories BPE files) fetched from the pinned revision and matched exporter hashes; no weights downloaded in that screen. Markdown file/heading links resolve, shell blocks pass `bash -n`, and `tools/check.sh` passes. Full downloads, exporter recipe and Linux/Mac fresh-clone verification remain pending.

- **TinyStories text input:** encode with the artifact tokenizer; reject empty, padding and multi-token strings rather than truncating. The pinned upstream tokenizer and Rust agree on three representative inputs; CUDA baseline/alias/ID/text output matches for a three-row request, with decode matching too. [Screen](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-text-input). CLI tests (12 passed; artifact-dependent TinyStories test also run), feature Clippy, full XSum/WMT14 corpus-encoding retest and `tools/check.sh --cuda` pass. Metal and performance evidence remain pending.

- **README and release-note draft:** updated the profile matrix and linked scoped evidence; release and fresh-clone gates remain pending. Corrected the earlier TinyStories fixed-start claim: the model accepts one non-padding prefix ID; CUDA seed 42 with `[[2]]` matches baseline/alias/unified and differs from `[[0]]` at 639/641 codes ([screen](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-prefix-correction)). README/draft relative links resolve and `tools/check.sh` passes; local tag and GitHub release `v0.1.0` are absent.

- **Sudoku provenance comparison:** searched every local and Mac Git blob and 2,171 bounded local files without locating the historical exporter. A [tracked verifier](tools/reference/compare_artifact_provenance.py) directly compares every non-provenance payload byte and verifies hashes, inventories and only the specified provenance change; it passed for both retained medium/hard re-exports. Six verifier tests and `tools/check.sh` pass. [Details](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#byte-level-comparison-of-alternate-exports). Neither alternate manifest is approved.

- **Reproducible CLI screen:** [CUDA report](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) records a tracked runner passing nine fixed profile requests and four unit tests including negative controls; `tools/check.sh` passes (537 report tests). A read-only Mac checkout/unreachable-blob search did not find the missing historical Sudoku exporter; no Mac files changed or MLX test run.

- **Linux export reproducibility (investigation only):** [report](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md) records seven exact manifest reproductions and the two Sudoku medium/hard provenance-only mismatches. A project-local Python 3.14.9 build-tree interpreter enabled old exporters; installation failed on `_ssl`, and no general Python setup or fresh clone is claimed. `tools/check.sh` passes. Original artifacts and pins unchanged.

- **LM1B rotary sidecar and help:** [CUDA screen](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md): default sidecar, override and baseline outputs match for small/medium/large with and without decode; missing and wrong-hash sidecars are rejected. CLI tests (11 passed), CLI Clippy and `tools/check.sh --cuda` pass. `ncm --help` and `ncm sample --help` describe current options and experimental paths; `tools/check.sh` also passed. Metal, fresh-clone setup, and text-input decision are pending.

- **Phase 5 text input:** XSum 11,334/11,334 and WMT14 3,003/3,003 Rust-encoded padded prefixes match pinned reference rows; first-row CUDA generations match ID input both with and without decoding. [Evidence and reproduction](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md). CLI tests (10 passed, corpus test separately run), CLI Clippy and `tools/check.sh --cuda` pass. TinyStories/LM1B `--text` and Metal remain pending.

- **Phase 5 text decoding:** CUDA CLI tests (10 passed), CLI all-target Clippy and `tools/check.sh --cuda` pass, including 18 real CUDA operator cases. [CUDA decode screen](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md) matches evaluator decoding on one retained generated row for each text profile, with codes/IDs unchanged. Corpus-wide and Metal evidence remains pending.

- **Phase 5 CLI routing:** CLI binary tests (9 passed) and all-target Clippy with `cuda-cudnn`; `setpriv --inh-caps=-all ... ./tools/check.sh --cuda` passes, including 18 real CUDA operator cases. [CUDA screen](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v1.md) compares baseline, alias and unified output for one request per profile. CLI link/test required `LIBRARY_PATH=$PWD/.reference/phase4-cudnn-probe-lib` and `LD_LIBRARY_PATH=$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib`; the first attempt without those failed to link `-lcudnn`. Metal and performance checks are pending.

- **Phase 5 plan (docs only) and cleanup:** wrote `docs/PHASE5.md`. Deleted untracked data not needed for Phase 5 on both
  machines: raw reports, superseded artifacts, build caches, old `.reference` snapshots and logs. Afterwards
  `tools/check.sh --cuda` and `tools/check.sh --mlx` pass, and the LM1B CLI samples and decodes on both GPUs.
- **Phase 4 audit (docs only):** [audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md) maps every
  profile to tokenizer, teacher-forced, end-to-end, quality and performance evidence on both GPUs; all relative links
  resolve. LM1B tokenizer, codec and license bytes are identical across sizes. Default `tools/check.sh` passes.
- **Metal, all LM1B sizes (no code changes):** Mac at `30349ae`: `tools/check.sh --mlx`, five real-Metal sampler tests
  (with `lm1b-experimental`) and feature Clippy pass. Small output is byte-identical before and after the memory fix
  (`0a3ce34` vs `30349ae`, batches 1/4/16, both samplers). Medium and large Metal teacher traces pass 51/51 against
  the strict CUDA references with zero argmax and ID mismatches. The CLI generates at all three sizes. 80-seed Metal
  campaigns pass: medium perplexity +0.17 [-0.39, 0.71], entropy +0.00068 [-0.00036, 0.00171]; large +0.10
  [-0.39, 0.60], entropy +0.00087 [-0.00030, 0.00205]; Python rescoring identical, summaries replay. Metal timing passes
  median, p95 and RSS in every cell for small, medium and large ([Metal report](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md)).
- **Medium/large CUDA quality and timing (no code changes):** 80-seed campaigns pass. Medium: perplexity
  +0.07 [-0.48, 0.61], entropy +0.00026 [-0.00084, 0.00137]. Large: perplexity +0.32 [-0.18, 0.81], entropy
  +0.00080 [-0.00040, 0.00199]. All 160 scores per size are `upstream_exact`, and summaries replay byte for byte.
  Timing: Rust median is 16/9/4.5% faster for medium at batches 1/4/16 and 12/6/5% faster for large at 1/4/8, with
  lower RSS and GPU memory. Pinned Python runs out of memory at large batch 16 on the 8 GiB card, so the large
  sweep uses batch 8 (the campaign batch); the failed run is retained.
  `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes with the GPU idle (530 report tests, including
  `test_real_writers_and_owned_process_samples_compose`, which had failed only while the GPU was busy).
- **LM1B medium/large step 2 (CUDA parity):** `Lm1bSize` (profile, config hash, layers, heads, width) drives
  the teacher, CLI (`ncm sample-lm1b`), and CUDA whitelists (efficient attention heads 16/20, fused rotary
  12/16/20). Both CUDA backends pass 51/51 teacher-forced tensors with zero argmax and ID mismatches
  at all three sizes against matching references: the cuDNN backend against strict, and the efficient backend
  against evaluator settings (default attention dispatch, convolution TF32). The efficient path against strict fails by
  design: level-zero reconstruction equals PyTorch with convolution TF32 exactly.
  `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes.
- **OWT coverage:** `ncm_core::OwtConfigOnly` validates `tests/fixtures/owt-config-v1.json` (pinned upstream
  YAML) and always refuses execution. `ncm-io` rejects OWT profiles, and `owt_shapes.rs` checks area/linear
  resampling, RoPE and masked attention at OWT's 1,025-position, 16-head shape.
- **Campaign and timing by size:** `campaign.py --size`, both cohort runners and the CUDA timing screen
  (`run_lm1b_small_capacity_screen.py --size`) accept all three sizes. A small rehearsal still reproduces the
  archive exactly, and a medium rehearsal completes every stage. The default check passes. Medium and large reuse
  the pre-registered seed lists, which are fresh for those models.
- **LM1B medium/large step 1:** the exporter takes `--size`; medium and large artifacts (manifests `2fea4a25…`,
  `86676fed…`) are deterministic across two exports, and `ncm-io` has separate per-size budgets plus a new test.
  The generalized CPU probe loads all three sizes. `setpriv --inh-caps=-all ./tools/check.sh` passes.
  Next: make the teacher read layers, heads and width from config; audit CUDA fused kernels for width
  assumptions; extend the trace tools, campaign and CLI by size.
- **Metal RSS fix (user chose "try one targeted fix"; since verified on Metal, see top entry):** host FP64
  penalties now read decoder logits through `visit_f32_rows` in blocks of two rows
  (`LM1B_PENALTY_ROWS`). On MLX each block is a separate exact GPU copy (multiply by one), so the CPU
  never touches the full logits buffer. The default (CUDA/CPU) implementation slices one host copy.
  Values and order are unchanged. Hypothesis being tested: Rust RSS growth equals the CPU-touched
  decoder logits. Checks: a new CPU row-visit test passes, and `setpriv --inh-caps=-all ./tools/check.sh --cuda`
  passes. A Metal row-visit test was added to `lm1b_categorical_fp32.rs`.
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
- `.claude/settings.json` (25 read-only Claude Code permission rules) is committed at the user's request. JSON validated; no code changed.

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

