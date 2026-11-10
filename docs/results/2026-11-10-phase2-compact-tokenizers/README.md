# Phase 2 compact tokenizer structures — Linux revalidation

**Linux passes all 78 comparisons. Apple validation of these new structures is
pending; Phase 2 is not closed.** No pins, assets, goldens, numerical budgets or
acceptance thresholds changed. No new dependency or project-local unsafe code.

## Measurements and representation choices

SSH access was restored; Mac main was inspected clean and safely pulled through
`2c195ed`. Other desktop work was left running. Full `tools/check.sh --mlx` passes.
The preceding tokenizer read-buffer fix improves Qwen to **86.97 ms / 455,245,824
bytes HWM**, versus fresh Python **145.97 ms / 288,587,776 bytes**. This remains a
memory failure, not acceptance. Raw unprofiled and allocation/VM diagnostics are
in `raw/apple-buffer-diagnostics.tar.gz`; instrumented runs are not timings.

Following the request to use more memory-efficient Rust structures:

- TinyStories stores immutable token bytes in **one arena plus u32 offsets**,
  instead of `Vec<Vec<u8>>`. For the pinned 8,192-token vocabulary: 46,185 payload
  bytes are unchanged; descriptors shrink **196,608 → 32,772 bytes**, eliminating
  8,192 persistent token allocations. These are representation counts, not RSS.
- Merge output ID is exactly `258 + rank`, validated during construction. Store
  one u32 output ID, not a redundant `(usize rank, u32 ID)` tuple. Ascending ID
  comparison preserves ascending rank and deterministic selection. Pre-size the
  bounded merge map. Temporary reverse lookups borrow arena slices rather than
  clone all token bytes; merge validation compares iterators without concatenation.
- Valid UTF-8 decode output moves its allocation directly into `String`; malformed
  byte sequences retain the identical replacement-character conversion. The first
  packed prototype still copied valid output and failed controlled decode timing
  (median ratio **1.05735**, interval upper **1.07670**); that negative evidence is
  retained, not waived. The ownership fix removes this measured deficit.
- HF Tokenizers 0.22.2 `get_vocab_size(true)` clones the complete base vocabulary
  just to count the union with added tokens. Count that exact union using borrowed
  model lookups and the small borrowed added-vocabulary map instead. Tests include
  overlapping/duplicate added tokens; no vocabulary validation was removed.
- The pinned BPE deserializer buffers tuple/legacy merges through an untagged serde
  tree. Authenticated GPT-2/Qwen canonical payloads use tuple merges exclusively.
  A strict private canonical deserializer builds the **same pinned BPE** through
  its public builder without the intermediate generic tree. The standard complete
  `TokenizerImpl` visitor still loads all preprocessing/postprocessing/added-token
  state. The wrapper is moved into the unchanged runtime model, not used as a new
  inference algorithm. Unsupported legacy/unknown model forms fail explicitly.
  Tests compare complete serialized state, options and invalid configurations.

Source research: pinned Tokenizers 0.22.2 `models/bpe/serialization.rs`,
`models/bpe/model.rs`, `tokenizer/mod.rs`, `tokenizer/added_vocabulary.rs`; existing
Rust byte-BPE ownership and measured allocation stacks. Upstream dependency
licenses remain unchanged; no fork of its inference algorithms.

## Linux results

Full `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4
./tools/check.sh --cuda` passes, including packed-entry bounds, invalid UTF-8,
ranked/all-occurrence merges, complete HF state, overlap-count and report mutation
regressions. All real tokenizer goldens/rejections pass in a cleared environment
without Python. Codec execution is unchanged from the previously qualified
642-trace/2,500-decision shared candidate; new benchmark outputs/logits also pass.

Fresh three-way, three-process matrix, five warmups/thirty trials:

| Profile | Old / new / Python warm load (ms) | Old / new / Python max HWM (bytes) |
| --- | --- | --- |
| TinyStories | 3.8445 / 3.4257 / 29.2385 | 12,677,120 / 11,120,640 / 73,916,416 |
| Qwen | 223.2246 / 89.8992 / 196.7840 | 194,494,464 / 81,559,552 / 240,099,328 |

All **78** unchanged performance/memory comparisons pass in
`raw/phase2-linux-compact-tokenizers-v2-acceptance-v2.json`. Fresh fixed 30-triplet
TinyStories and BERT controls resolve timing uncertainty. TinyStories candidate /
previous encode/decode/skip-special medians are **0.99983 / 0.99614 / 1.01329**;
paired-process 95% interval upper bounds are **1.00400 / 1.00584 / 1.02249**.

Codec checked-load three-process envelopes also varied despite unchanged codec
code. Nine additional full-codec triplets, all retained, yield candidate/previous
median **0.95809**, interval **[0.94424, 0.99002]**. The explicit `codec` load control
validates every complete benchmark report's outputs/scopes before applying the
same paired-process bootstrap. It does not reuse lifecycle timings as warm loads.
All intermediate failed audits and measurements are archived. No fastest-run
selection or changed thresholds; fresh processes remain the independent units.

## Profile-capacity follow-up

Candidate `45743d0` passes full Metal checks and real tokenizer goldens. Apple Qwen
improves further to **65.82 ms / 326,828,032 bytes HWM**, but still exceeds the
matched Python memory band. This negative diagnostic is retained in
`raw/compact-qwen.json`; it is not relabeled as an acceptance pass.

The canonical parser now reserves vocabulary and merge containers from the pinned
profile capacities (GPT-2 50,257; Qwen 151,643), avoiding geometric map/vector
construction growth. These are authenticated, compile-time allocation bounds, not
untrusted JSON size hints. Tuple count is bounded by base vocabulary capacity;
new tests cover exact capacity and rejection above it. Runtime structures and
model algorithms are unchanged. This follows the compact-ownership work rather
than introducing an alternate global allocator or allocator-trimming benchmark.

This follow-up returned through Linux first: full CUDA checks, real tokenizer
goldens/rejections, a fresh full matrix, and fixed 30-triplet BERT/GPT-2/TinyStories
controls pass **78 comparisons** in `raw/phase2-linux-capacity-tokenizers-acceptance-v2.json`.
The unchanged-codec nine-triplet load control is reused with exact source binding;
its data are not claimed as newly collected. Initial uncertain audit is retained.
The new config is `raw/phase2-linux-capacity-tokenizers-config.json`. Apple must
still test this latest allocation-bound candidate.

## Reproduction and next steps

`evidence.json` hashes every raw archive/log/report. Unpack `raw/linux-NAME.tar.gz`
to `reports/phase2-performance-linux-NAME/`. The final config records its inputs;
qualification/lifecycle paths refer to preceding unchanged-codec evidence from
the shared-loading report. Run:

```sh
python3 tools/bench/audit_phase2_performance.py \
  docs/results/2026-11-10-phase2-compact-tokenizers/raw/phase2-linux-compact-tokenizers-v2-config.json \
  reports/compact-reproduced.json
```

A new counterbalanced Apple runner records CPU activity and explicitly **does not
claim CPU affinity** on macOS. The optional Apple control policy retains all
processes and the same bootstrap/bands; missing activity telemetry fails closed.
New native/adapted-Python MLX lifecycle harnesses reuse the established four phases
and output-identity checks. These tools still require real Apple execution.

Next: safely pull this reviewed candidate on Mac, validate compact structures and
memory, full trained Metal traces, repeated load/request stability, matched complete
three-way matrix and separate lifecycle evidence. Return further shared production
changes through Linux first. Historical worktrees and `.pi/` remain untouched.
