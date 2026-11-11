# Additional Phase 2 — initial lower-risk Linux candidates

**Provisional Linux evidence, not final acceptance.** This follows the
[frozen/refreshed baseline](../2026-11-11-phase2-optimization/README.md) and approved
[five-step plan](../../PHASE2-OPTIMIZATION-REVIEW.md). Three candidates were examined
in sequence: HF ID-only encoding, TinyStories in-place merges, then reconstruction
latent reuse. No pins, goldens, numerical budgets, output counts or authentication
checks changed. Apple testing has not started; both final runtime gates are pending.

The [experiment ledger](../2026-11-11-phase2-optimization/experiments.json) also
tracks prepared state, loading copies and conditional higher-cost ideas. Those
requirements are not silently treated as implemented or completed.

## Diagnostic allocation attribution

The new `asset_allocations` example wraps **System only in that diagnostic
executable**. Normal library/benchmark/inference allocators are unchanged. The
narrow `tests/support/allocation_counts.rs` unsafe implementation forwards original
layouts/pointers without touching their contents; instrumentation uses atomics,
not allocations or logging inside allocator callbacks. Measurement scopes serialize
and disable recording on unwind. Two explicit example tests check allocation,
reallocation, freeing, retained contents and unwind recovery.

It counts successful **Rust GlobalAlloc requests**, not C/native-library allocations,
allocator-internal copies, RSS, live storage or physical peaks. Returned outputs
stay alive through the end of each region; validation, hashing and their destruction
are excluded, as in the wall-time workload. Parallel tokenization is disabled.
These instrumented runs are never used as latency acceptance evidence.

Median counts per 89-string encode trial (three fresh processes, five warmups,
thirty measured trials/process):

| Profile | Accepted allocations | HF fast only | HF fast + in-place Tiny |
| --- | ---: | ---: | ---: |
| BERT | 21,333 | 19,989 | 19,989 |
| GPT-2 | 21,502 | 21,502 | 21,502 |
| Qwen | 21,180 | 21,180 | 21,180 |
| TinyStories | 6,861 | 6,861 | 6,602 |

BERT loses **1,344 allocation requests (~6.3%)**, but requested allocated bytes
only decrease 3,504,436 → 3,501,656. TinyStories loses **259 requests (~3.8%)**, with
requested allocated bytes 353,564 → 349,368. These are modest allocation benefits,
not equivalent RSS reductions. BERT decode still makes 19,876 Rust allocations
per trial; its pinned per-token WordPiece cleanup merits later investigation if
profiling justifies a compatible change.

## Tokenizer experiments

`TextTokenizer::encode` now calls the pinned HF `encode_fast` with special tokens
still enabled; only unused offset work is omitted. A new wrapper regression checks
normalization, accents/Unicode, added specials, postprocessing, truncation, padding
and the request bound. Real-profile goldens and mutation rejection probes pass.

The isolated HF experiment's native median encode ratios versus accepted Rust are
BERT **0.9815**, GPT-2 **1.0038**, Qwen **0.9852**. These small three-process changes
are insufficient for a general speedup claim; GPT-2/Qwen allocation counts are
unchanged. Keep the no-benefit evidence, not just the BERT result.

TinyStories next replaces the freshly allocated merge output vector with stable
left-to-right compaction of the existing word buffer. The write cursor never
passes unread symbols; each round still merges all nonoverlapping occurrences of
the lowest-ranked pair. No regex, vocabulary arena, merge ranks or byte decoding
changed. Differential tests cover all **2,047 binary strings through length ten**,
three long overlapping/repeated inputs, and the retained special/Unicode fixtures.

The subsequent matched TinyStories encode median is **0.236233 → 0.235164 ms**;
no material latency gain is established. The small allocation reduction is the
measured mechanism. Separate intermediate patches, raw output digests and source
hashes are retained, so combined results do not conceal either isolated experiment.

## Shared reconstruction reuse

Untraced `Codec::reconstruct` now runs encode once and feeds its final quantized
latent to the shared decoder tail. It returns both snapshot collections by cloning
immutable tensor handles, which the existing `WeightBackend` contract explicitly
permits. No snapshots/logits/IDs disappear; no mutable accumulator is shared and no
permanent cache/workspace is introduced. Standalone arbitrary-code decode keeps
its full code-to-latent construction.

`reconstruct_traced` deliberately retains the independent decoder construction so
all existing upsample/contribution/cumulative observations remain available. Both
routes use the same final normalization/projection/selection code. Traced stage
timings therefore do **not** represent optimized reconstruction wall time.

Tiny CPU/CUDA tests and trained captures now check **every untraced snapshot after
another decode**, not only final logits/IDs. The trained capture requires bit-exact
agreement with the traced route before emitting its report. All **642 traces and
2,500 decisions** then pass the unchanged CUDA oracle qualification. This is
codec reconstruction/reference parity, not generation or Sudoku-solving quality.

Matched uninstrumented reconstruction medians, milliseconds:

| Batch | Accepted Rust | Reuse candidate | Qualified Python | Candidate/accepted |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.0003 | 2.2853 | 4.6748 | 0.7617 |
| 2 | 4.0104 | 2.6712 | 5.0842 | 0.6661 |
| 4 | 5.1257 | 3.3667 | 5.8467 | 0.6568 |

Separate fresh CUPTI diagnosis (checked load + two B1 reconstructions) drops from
**1,312 → 918 kernels**, driver events **33,746 → 24,418**. Summed kernel activity
falls 4.464 → 3.084 ms; these overlapping/instrumented sums are not request latency.
The source change and count reduction explain the work removed without relying
on the prior baseline's misleading identical-binary B1 timing difference.

## Gates still open

The full current Linux diagnostic audit is **74/78**, not an acceptance pass.
All pooled median/p95 points and comparable host-memory ratios pass. Four native
process envelopes remain above 1.05:

- B1 resident decoder block: **1.10338** (point median ratio **1.00096**).
- Qwen decode: **1.08123** (point median **0.98027**).
- BERT checked load: **1.09601** (point median **1.00906**).
- GPT-2 checked load: **1.05581** (point median **1.00000**).

Do not suppress these failures or treat them as regressions proved by source
changes: the small baseline sample is uncertain. Matched controlled follow-ups
are required under unchanged bands before the Linux gate can close. No identity
control substitutes for changed tokenizer executables. The initial HF experiment's
separate uncertain cells remain in its raw reports.

A fresh nine-process lifecycle report has complete phases and identical solved-board
output identities. `tools/check.sh --cuda` passes, including 29 report-tool tests;
17 tokenizer unit tests and the two allocation-example tests pass separately.
Cleared-environment tokenizer probes pass 356 encode and 2,314 decode comparisons
and four rejection mutations per profile. No new MLX evidence is implied.

## Evidence and reproduction

[evidence.json](evidence.json) pins source files, compact raw archives and
[experiments-summary.json](experiments-summary.json). The latter holds complete
intermediate tokenizer summaries and allocation count ranges/medians, not just
highlighted improvements. All source changes are narrow; no dependency was added.

Key commands (new report/output paths must not already exist):

```sh
cargo test -p ncm-tokenizers --locked
cargo test -p ncm-tokenizers --example asset_allocations --locked
CARGO_BUILD_JOBS=4 cargo build -p ncm-tokenizers --release --locked \
  --example asset_bench --example asset_probe --example asset_allocations
TOKENIZERS_PARALLELISM=false taskset -c 2 target/release/examples/asset_allocations \
  artifacts/canonical/tokenizers-qualified tests/fixtures/tokenizers-v1.json bert NEW_REPORT.json
# Repeat each profile in three fresh processes before and after each experiment.
python3 tools/bench/run_phase2_linux_comparison.py \
  .reference/phase2-optimization-baseline .reference/phase2-optimization-baseline-bin \
  NEW_THREEWAY_DIRECTORY --workloads bert gpt2 qwen3 tinystories --cpu 2
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
# For reconstruction: rebuild CUDA codec_bench/codec_profile/codec_trace/codec_lifecycle,
# capture/qualify with the same pinned oracle commands as the baseline report,
# then run the three-way runner with all default workloads, --cpu 2.
```

Exact executable identities, commands, process order, hardware and telemetry are
in each archived `environment.json`; lifecycle commands are embedded in its JSON.
`raw/hf-fast.patch`, `raw/tiny-compaction.patch` and
`raw/reconstruction-reuse.patch` preserve independent changes relative to `3307dff`.
The allocation harness source is committed alongside this report and fingerprinted
in each diagnostic. The saved baseline diagnostic executable remains local in
`.reference/phase2-optimization-profile-baseline-bin/`; no executable is published.

To reproduce the full diagnostic audit without GPU/weights/reference environments:

```sh
D=docs/results/2026-11-11-phase2-low-risk
mkdir -p reports/phase2-optimization-reuse-v1
tar -xzf "$D/raw/reconstruction-threeway-v1.tar.gz" -C reports
cp "$D/raw/qualification.json" "$D/raw/lifecycle.json" reports/phase2-optimization-reuse-v1/
python3 tools/bench/phase2_summary.py reports/phase2-optimization-reuse-threeway-v1/previous /tmp/reuse-previous.json
python3 tools/bench/phase2_summary.py reports/phase2-optimization-reuse-threeway-v1/candidate /tmp/reuse-candidate.json
python3 tools/bench/audit_phase2_performance.py "$D/raw/diagnostic-config.json" /tmp/reuse-audit.json
```

The audit must exit 1 and reproduce **74/78 with four pending cells**. Compare the
summaries/audit to committed JSON. Extract `raw/tokenizer-experiments-v1.tar.gz`
under `reports/` for every intermediate tokenizer/allocation report. Large trained
tensors and licensed weights remain outside Git. Phase 2 remains open.
