# Further Phase 2 optimization review

Research date: 2026-11-11. Reviewed repository: `3307dff`; accepted production
candidate: `083c1e5`. **The user approved the five-step plan below on 2026-11-11
and requested its commit/push before implementation. Phase 2 is reopened for this
additional work; implementation and new runtime acceptance remain pending.**
This planning change contains no production changes, new GPU measurements or
Phase 3 implementation.

## Recommendation

There are credible opportunities worth a bounded optimization round before
Phase 3. Passing Python non-inferiority establishes a useful baseline, not an
optimal implementation. The strongest findings are redundant quantizer work in
reconstruction, MLX's per-position graph construction, and repeated immutable
metadata construction. CPU tokenization and checked loading have smaller,
separately testable opportunities.

Start with existing-library APIs and removal of redundant work. Escalate to
custom kernels or graph replay only when new profiles justify their maintenance
and memory costs. **No additional speedup percentage is established by this
review.** Static operation counts and byte estimates below are not benchmarks.

The earlier accepted Phase 2 result remains valid for its recorded candidate.
The approved additional work must now be completed before Phase 2 closes again;
Phase 3 remains unstarted and outside this authorization.

## What the existing measurements actually establish

Sources: [final acceptance](results/2026-11-10-phase2-acceptance/README.md),
[Linux summary](results/2026-11-10-phase2-acceptance/phase2-final-linux-candidate-summary.json),
[MLX summary](results/2026-11-10-phase2-mlx-remediation/phase2-mlx-remediated-candidate-summary.json).
All times below are existing pooled medians, milliseconds.

| Current Rust scope | CUDA B1 | MLX B1 |
| --- | ---: | ---: |
| Encode | 2.0665 | 13.1570 |
| Decode | 1.2429 | 11.9468 |
| Reconstruction | 3.0551 | 24.9826 |
| Resident decoder block | 0.1783 | 1.1006 |

Reconstruction B1/B2/B4 is **3.0551/4.0014/5.1322 ms** on CUDA and
**24.9826/27.0685/29.2383 ms** on MLX, rounded from the summaries. B4 takes about
1.68 times B1 latency on CUDA and 1.17 times on MLX: throughput grows about
2.38 and 3.42 times, respectively. This is consistent with substantial fixed
costs, but does not alone distinguish CPU dispatch, launch overhead, or GPU
underutilization. These different machines are not a controlled hardware or
language comparison. Separate scope medians are not additive decompositions.

The [retained CUDA profile](results/2026-11-10-phase2-cuda-remediation/profiles/codec-remediated-cupti.json)
still contains 1,312 kernels. Its 4.451 ms summed kernel duration and 125.698 ms
summed driver-event duration are **instrumented diagnostics**, not request wall
time; spans can overlap. Gather/layout/allocation/launch work remains worth
profiling, but those sums do not quantify the attainable speedup.

Current warm codec load is about 195.9 ms CUDA / 180.0 ms MLX, with maximum host
HWM about 606.4 / 293.9 MB. Apple Qwen is already about 63.7 ms / 69.8 MB.
The former multi-gigabyte Apple allocator-growth problem has been fixed; it
should not be presented as a remaining defect. MLX active allocations return to
zero after model drop in the qualified probes.

Existing evidence does **not** establish current per-request CPU cycles,
allocation counts, GPU bandwidth/occupancy, or energy. Add focused diagnostics
before claiming improvement in those dimensions. Higher reported GPU utilization
is not itself the objective; less work and lower time/energy per useful output
can reduce utilization.

## Ranked opportunities

### 1. Shared prepared state: low-risk first experiment

Files: `crates/ncm-model/src/{codec,blocks,lib}.rs`,
`crates/ncm-backend-cuda/src/{codec,resample}.rs`.

- Use the validated positional embedding table directly instead of uploading
  `0..length` and gathering the same entire table on every encode, where shapes
  permit. Otherwise prepare the position indices once.
- Prepare immutable codebook squared norms. Current `codebook_distances`
  recomputes `sum(book²)` at every level on every encode.
- Prepare resampling indices, weights, and area-bin metadata once per validated
  shape instead of rebuilding/uploading them on each call.
- Resolve hot weight names into typed level/block records after strict loading;
  retain the existing named inventory/API where needed. Avoid repeated string
  formatting and tree lookups. Measure their cost before expecting a large gain.
- Make trace-label construction genuinely conditional for `NoTrace`, rather than
  assuming the optimizer removes heap-backed formatting through fallible calls.

For the trained twelve-level codec, linear metadata, padded-area metadata, and
codebook norms need approximately **25,680 logical bytes** in the proposed
representation, excluding tensor objects, allocator rounding, and layout copies.
This is a bounded cache, not justification for arbitrary permanent workspaces.
Keep model-derived state model-owned, tied to its backend/device and weights;
verify drop behavior and isolation between models. Do not replace the artifact
loader's validation structures merely to change map types.

**Numerical trap:** do not drop `sum(x²)` from distance selection just because it
is constant across a row in real arithmetic. FP32 rounding can change ties and
IDs. Preserve operation ordering and first-index tie semantics.

### 2. Reconstruction reuse: strongest shared algorithmic opportunity

`Codec::reconstruct_traced` encodes, then calls `decode_device` on the generated
codes. Encode already computes every selected embedding's upsample, refinement
contribution, and cumulative sum. Decode repeats that construction before
running the decoder block.

Prototype a shared internal quantizer result that lets reconstruction reuse the
already computed latent and cumulative states. Keep independent public encode
and arbitrary-code decode behavior unchanged. This can eliminate redundant
convolutions, interpolation, additions, allocations, and dispatches on both GPUs.
It is not equivalent to deleting the whole decode cost: the decoder block,
final normalization/projection, output selection, and required outputs remain.

The output contract matters:

- `Encoded` and `Decoded` expose per-level snapshots. Their logical storage totals
  **2,187,264 bytes at B1 / 8,749,056 at B4** if both sets are separately stored.
  Actual allocator use and physical sharing differ by backend.
- Sharing immutable storage may be valid; aliasing must not let later work alter
  any retained snapshot, including on lazy MLX. Do not introduce in-place updates
  or silently remove outputs to obtain a benchmark win.
- Preserve all trace names/values. Retaining extra trace-only intermediates may
  cost memory; an optimized untraced route must still be checked against traced
  execution and standalone `decode(encode(...))`.
- Validate all batches and input classes. Mathematically identical reconstructions
  are not automatically bitwise identical under different layouts/kernels.

### 3. MLX resampling and execution: largest platform-specific opportunity

Files: `crates/ncm-backend-mlx/src/{ops,codec}.rs`.

`area1d` creates a slice/reduction for each output bin. `linear1d` creates two
slices, two multiplies and an add for each output position, then concatenates.
For levels `[1,2,4,6,9,13,18,26,37,50,67,89]`:

- Encode builds 232 middle-level area bins.
- Encode or decode builds 890 linear output bins.
- Reconstruction builds 1,780 linear bins: **8,900 slice/multiply/add source
  calls**, excluding concatenations and scalar constructors.

These are **not Metal kernel counts**: MLX is lazy and may optimize execution.
Nevertheless, the host graph construction is real and structurally avoidable.

First prototype vectorized gathers/reductions with prepared geometry, building
on the CUDA approach. Measure padded-gather temporary memory as well as latency;
a compact direct kernel may eventually be preferable if padding dominates.
Then test existing pinned MLX fast RMSNorm/LayerNorm, followed by compilation of
pure, bounded-shape subgraphs. Measure graph construction separately from
synchronized execution, without using profiling runs for acceptance.

Important constraints:

- Current MLX interpolation constructs coordinates in host `f64` before casting
  weights to FP32; the qualified CUDA implementation uses FP32 coordinates.
  A retained FP64-coordinate CUDA PoC failed expanded comparisons. Do not blindly
  unify these paths: independently qualify the proposed MLX arithmetic against
  the unchanged oracle and expanded resampling cases.
- Compilation can fuse operations and change rounding. Existing trained errors
  have approached the tolerance boundary; no tolerance waiver is available.
- Pinned mlx-rs compilation documentation warns that arrays must be explicit
  function inputs; capturing arrays in compiled closures can cause undefined
  behavior. Preserve purity, lifetimes, and explicit shape specialization.
- Record first-call compilation, specialization/cache memory and steady state.
  Avoid an unbounded compiled variant for every possible request shape.
- Fast attention is a secondary experiment, not the first priority. Native MLX
  0.32.2 full-attention dispatch includes head width 64, which this codec uses,
  but actual FP32 dispatch, rounding and speed still require hardware testing.
  Preserve bidirectional attention, Q/K RMS normalization, and exact-erf GELU.

### 4. CUDA launch, layout and temporary-memory reduction

After shared changes, capture a new profile: an old hotspot may disappear.
Prioritize measured residual costs among:

1. Repeated contiguous conversions and weight transposes. Prepare a required
   layout once only if the saved copies justify persistent storage; do not keep
   duplicate copies of every weight speculatively.
2. Deterministic selection. Current CUDA uses value reduction, comparison,
   index construction, masking, then index reduction. A narrowly audited
   value/index reduction could preserve first-index ties with fewer launches.
   Native Candle argmin alone is not a valid replacement: its tie behavior was
   already rejected. Include wide axes and signed-zero ties.
3. Direct linear/area resampling kernels if composed gathers remain expensive.
   Preserve qualified coordinates, bin boundaries and reduction behavior.
4. Convolution layout/im2col traffic. Pinned Candle's non-cuDNN conv1d path uses
   im2col plus matmul. Measure the actual traffic before considering alternative
   algorithms or adding cuDNN/build dependencies.
5. CUDA graph replay only after allocations, layouts and shape plans are stable.
   cudarc 0.19.10 exposes capture/replay, but capture safety, synchronization,
   allocation lifetime, stable buffers and thread-confined/serialized graph
   access need a PoC; do not add unsafe Send/Sync to bypass graph restrictions.
   Replay must not overwrite previously returned results/snapshots. Include
   capture/setup cost and break-even request count.

Do not increase the existing 128 MiB pool release threshold as the default next
optimization. It already solved a measured allocation tail; more retained memory
is not automatically faster. It is neither a hard pool limit nor live model data.

### 5. Tokenizers: low-risk API change, then compact algorithms

Files: `crates/ncm-tokenizers/src/{assets,bpe,hf_bpe}.rs` and pinned Tokenizers
0.22.2 source.

**HF encode:** the wrapper returns IDs only but calls `encode`, which computes
offsets. The pinned dependency already provides **`encode_fast`**, skipping
offset computation. Test this first, retaining special tokens, normalization,
postprocessing, padding/truncation and exact IDs. It needs no dependency upgrade.

**TinyStories byte BPE:** the current loop rescans every adjacent pair for the
lowest rank, then allocates/rebuilds a new word vector per merge round. First try
stable in-place compaction or two reusable request-local buffers, preserving
all-occurrence merging. Next compare an array-backed linked-symbol structure
with a rank/position heap, as used by the pinned HF BPE implementation. Worst-case
rescan/rebuild work is quadratic in word length; a neighbor-update heap can
improve long-piece behavior but add overhead for ordinary short words.
Overlapping pairs, leftmost ties, all-occurrence semantics, special-token splits
and malformed UTF-8 decoding need differential tests, not assumed equivalence.

The immutable merge dictionary is another experiment: a CSR-like row-offset
table indexed by left token, with sorted right-token/output entries. At 8,192
tokens / 7,934 merges, logical storage is about **96,244 bytes with u32 entries**,
or **64,508 with checked u16 token IDs**. This is not measured HashMap savings or
proof of faster lookup. The public structural constructor admits up to 65,536
tokens, so bounds/sentinels need care. Reject a dense 8,192² u32 pair table:
**256 MiB** is disproportionate. Keep the existing compact vocabulary arena.

**BERT decode:** pinned WordPiece cleanup performs eleven replacement passes
per token when enabled. The qualified BERT decoder enables this; the outer
wrapper's separate cleanup flag is false for all four qualified profiles.
Profile the dependency's per-token allocations, not the inactive wrapper cleanup.
If material, consider a narrowly compatible decoder optimization or reviewed
upstream patch. Preserve cleanup order and per-token boundaries; whole-string
cleanup after joining is not equivalent. Any vendoring/pin change needs explicit
review, not a silent dependency edit.

**Cache caution:** HF BPE already has a bounded cache (default capacity 10,000,
maximum cached piece length 256) using a nonblocking lock. Repeating 89 fixture
cases measures a warm workload; add separately labeled unseen-text and mixed
workload tests before increasing caches or claiming general tokenizer speedups.
Do not replace the regex/Unicode engine without exhaustive semantic evidence.

### 6. Checked loading: reduce copies before adding parallelism

Files: `crates/ncm-io/src/lib.rs`, `crates/ncm-model/src/lib.rs`, backend uploads.

The authenticated bundle contains about **162.3 MB**, including 59.6 MB codec,
102.5 MB generator and 0.2 MB training safetensors. The generic `Artifact` owns
all files; codec construction then creates a temporary `Vec<f32>` for each
runtime tensor before backend upload. The largest such tensor is **23,330,816
bytes**. These are logical byte counts, not predicted RSS reductions.

First test a checked raw-byte upload path, retaining complete host preflight
before the first device allocation. Candle has `Tensor::from_raw_buffer`; inspect
alignment/endian handling rather than adding unchecked casts. MLX's existing
`Array::try_from(TensorView)` eagerly copies: it is not zero-copy. A native owned
array may still be necessary on unified memory.

A later, explicitly scoped codec loader could authenticate/validate unused
payloads without keeping every payload resident, retaining codec bytes through
upload. That is a new ownership/API design: do not change generic `Artifact::file`
behavior or skip generator/training validation in the current benchmark. Streaming
must preserve file and tensor hashes, limits, inventory/path checks and the
same authenticated bytes used for inference. An mmap reopened after validation
introduces mutation/lifetime hazards; mmap alone is not a secure zero-copy plan.

Parallel hashing/loading is lower priority: it may exchange wall latency for
more CPU and peak memory, especially under existing CPU-affinity controls.
Report CPU-seconds and allocated resources, not just elapsed load time. Do not
add a global allocator, trimming policy, or home-grown cryptographic primitive.

## Defer unless profiling changes the priority

- **Level-zero lookup:** only sixteen codebook entries make precomputed upsample
  and refinement tables plausible. One table is 1,458,176 logical bytes; both
  are 2,916,352. This trades persistent memory and setup work for compute and
  needs batch-sensitive numerical checks. It is not the first choice for a
  memory-efficiency request; reconstruction reuse can remove duplicated work
  without those permanent tables.
- **Snapshot-free output APIs:** potentially useful later, but a separately
  authorized contract/workload, not a transparent optimization of current APIs.
- **Attention-only rewrites:** the separately measured resident decoder block is
  a small part of current request latency. Prioritize quantizer/dispatch work.
- **More batching/concurrency:** existing results show throughput potential, but
  waiting to fill batches changes latency/service semantics. Do not substitute
  B4 throughput for a B1 latency improvement.
- **FP16/BF16/TF32, approximate GELU, altered codebooks or reduced validation:**
  outside this unchanged-FP32 proposal. No hidden precision/fidelity trade.
- **Generator KV caches, speculative decoding, full text-model kernels:** Phase 3
  or later work, unsupported by the current codec/tokenizer measurements.

## Approved five-step Phase 2 plan and decision gates

These are the five steps approved by the user, now part of Phase 2 rather than
optional Phase 3 preparation. Commit/push this planning revision first; then
implement and evaluate the bounded work. Step 4's measurement requirements apply
throughout steps 1–3 and any step 5 iteration, not only after accepting a change.
Within every iteration, Linux/CUDA implementation and validation finish before
Apple/MLX; both gates precede the next iteration. Any shared fix discovered on
Apple returns through Linux validation first. Ask the user about consequential
ambiguities, unavailable resources or required contract changes.

### 1. Freeze the accepted implementation as the new baseline

- Freeze repository `3307dff` (production behavior `083c1e5`) as current Rust;
  retain the unchanged qualified Python comparator and historical evidence.
  Preserve baseline binaries/worktrees without overwriting existing ones.
- Record source/executable/artifact/workload hashes, hardware, environments and
  harness identity. Refresh controlled current-Rust/Python measurements; do not
  compare a fresh candidate solely against old workstation timings.
- Lock the unchanged acceptance matrix and bands before production edits. Audit
  output work, timing/synchronization, cache state, fairness and reproducibility.
- Profile CPU/allocation and CUDA costs separately from acceptance runs. Separate
  host graph construction, GPU execution, loading and process lifecycle.

**Deliverable:** reproducible baseline manifest/reports, measurement configuration
and ranked hypotheses. Existing evidence is reused only where explicitly valid;
reused evidence is never described as a fresh measurement.

### 2. Prototype lower-risk changes independently

- Investigate prepared geometry/codebook norms/position embeddings and removal of
  unnecessary hot-path name/trace construction.
- Prototype reconstruction reuse without removing independent public snapshots,
  arbitrary-code decode support, or trace observations.
- Test pinned HF `encode_fast` and TinyStories request-local merge-buffer reuse.
- After the Linux gate in step 3, prototype vectorized MLX resampling and qualify
  its coordinate/reduction arithmetic independently.
- Track checked-loading copies and compact merge representations as additional
  candidates; prioritize them using profiles rather than assuming a benefit.

Each experiment gets a bounded hypothesis, isolated before/after evidence,
regression tests and a keep/reject decision before combining winners. Preserve
negative results. The candidate ledger must account for every ranked opportunity
in this review, including evidence-backed deferrals; approval does not require
shipping every idea or inventing a positive speedup.

### 3. Complete Linux/CUDA validation first, then Apple/MLX

- Validate shared/CUDA changes with `tools/check.sh --cuda`, unchanged primitive,
  component/tiny/trained qualifications, exact decisions and tokenizer probes.
  Close the full Linux performance/memory matrix against current Rust and Python
  before Apple implementation/testing of the candidate.
- Recheck Mac availability, competing work and remote state; announce the first
  MLX run. Pull reviewed pushed commits rather than overwriting its checkout.
  Profile host/Metal work, validate shared changes, then test Apple candidates.
- Run `tools/check.sh --mlx`, fresh trained qualification and cleared-environment
  tokenizer probes; close the same Apple matrix against current Rust and the
  qualified adapted Python MLX reference. Preserve Linux-first ordering for fixes.
- Include cache/model isolation, prior outputs surviving later requests, lazy
  snapshots, deterministic ties and all new algorithm-boundary regressions.

**Deliverable:** separate real-hardware correctness reports and fail-closed
acceptance audits for both platforms; CPU/default checks cannot substitute.

### 4. Measure latency, CPU work, memory, startup and teardown

- Collect median/p95/throughput, process-level uncertainty and comparable host
  live/peak memory for all existing codec batches/scopes and tokenizer profiles.
- Add CPU-time/allocation diagnostics and distinguish host construction from GPU
  execution. Record first-call/compile/capture cost, checked load, teardown,
  repeated load/drop and long-lived cache/workspace growth where applicable.
- Keep lifecycle and additional workloads separately labeled; do not change the
  existing matrix or hide costs in warmup. Missing hardware evidence is pending,
  not passing; unavailable extra telemetry must be disclosed without a claim.
- Publish raw favorable/negative runs, hashes, exact commands and limitations,
  plus clean-checkout reproduction of summaries/audits.

**Deliverable:** reproducible quality AND performance evidence, not an isolated
microbenchmark or a trade of lower latency for unexplained memory/CPU growth.

### 5. Escalate only for remaining measured bottlenecks; audit closure

Re-profile after lower-risk changes. Consider custom selection/resampling,
layout preparation, fast MLX normalization/compilation, compact merge lookup or
direct uploads only where the remaining cost justifies the experiment. Record
explicit go/no-go decisions for CUDA graphs, new loader ownership, decoder
adapters and level-zero tables, including memory/setup/maintenance costs.
Dependency/pin, trust/API, precision or tolerance changes require user review;
this approval does not silently authorize them. Every escalation repeats steps
2–4 in Linux-first order. No obligation to pursue speculative kernels.

Close Phase 2 only after a prompt-to-artifact completion audit covers all five
steps, every experiment disposition, unchanged correctness, both acceptance
gates, reproducible reports and updated status documents. If none of the proposed
changes provides a defensible benefit, retain accepted production behavior and
publish the complete negative/no-go evidence instead. A failed candidate is not
a passing gate: verify the actual retained final candidate on both platforms.
Phase 3 still requires separate authorization.

### Evidence checklist (pending runtime work)

Planned final evidence root: `docs/results/2026-11-11-phase2-optimization/`.
These are deliverables, not files or results claimed to exist yet.

| Approved requirement | Required evidence before closure | Status |
| --- | --- | --- |
| Step 1: freeze/refresh baseline | `baseline.json`, source/binary/artifact identities, matched raw baseline runs and locked comparison config | Pending |
| Step 2: independent lower-risk experiments | `experiments.json` with hypothesis, source diff, checks, raw measurements and keep/reject/defer reason for each opportunity | Pending |
| Step 3: Linux then Apple | Ordered check/capture/probe logs; `linux-acceptance.json` and `mlx-acceptance.json`, each covering all 78 comparisons against current Rust and Python | Pending |
| Step 4: all resource/lifecycle dimensions | Raw reports, CPU/allocation diagnostics, lifecycle/load/drop evidence, limitations and clean-checkout audit reproduction | Pending |
| Step 5: conditional escalation and final audit | Profile-linked go/no-go decisions; `completion-audit.md` mapping requirements to inspected final-candidate evidence | Pending |
| Publish and close Phase 2, not Phase 3 | Final `README.md`, evidence index, reviewed commits/push and synchronized plan/tracker/context status | Pending |

### Required checks and reporting

- Keep pins, artifacts, schemas, goldens and tolerances fixed. Preserve primitive,
  component, tiny/trained traces, all code/output decisions, near-tie evidence,
  snapshots, rejection tests and tokenizer goldens. Add regressions for prepared
  state ownership, old outputs surviving later requests, and new algorithm edges.
- Run `tools/check.sh --cuda`, then `tools/check.sh --mlx`, plus fresh trained
  qualifications and cleared-environment real tokenizer probes on each platform.
- Compare B1/B2/B4 encode/decode/reconstruct/block and all tokenizer scopes to
  **current accepted Rust as well as qualified Python**, not only the slower
  historical Rust. Keep median ≤1.05, p95 ≤1.10 and comparable host peak/live
  memory ≤1.05; do not widen bands to accommodate an optimization.
- Minimum three fresh processes, five warmups, thirty measured trials. Use
  rotating matched runs and the existing process-level uncertainty procedure;
  predeclare additional controls when necessary and retain every run.
- Separately report CPU time/allocation counts, first call/compilation, load,
  teardown, repeated load/drop, and long-lived cache/workspace growth. Additional
  workloads are explicitly separate from the unchanged acceptance matrix.
- Keep CUDA pool/Torch allocator counters distinct from device-wide usage; keep
  MLX active/cache counters distinct from host RSS/HWM. Do not add unified-memory
  counters to RSS as if they were disjoint physical storage. Energy requires
  suitable controlled measurements; missing telemetry remains missing.

**Stop rule:** passing non-inferiority is necessary, not evidence of a useful
optimization. Retain changes only with reproducible latency, CPU-work or memory
benefit and acceptable complexity. Reject candidates that fail parity, regress
another dimension, or yield gains indistinguishable from noise; retain accepted
production behavior where no candidate qualifies. A blocked resource, uncertain
gate or exhausted defensible path requires an evidence-backed status report and
question to the user, not a completion claim. Document attempted paths, blocker
and next input needed. Do not pursue speculative kernels indefinitely to force a
positive outcome, or proceed to Phase 3 without separate authorization.

## Sources and verification

Local source files are named above; dependency inspection used Candle 0.11.1,
cudarc 0.19.10, Tokenizers 0.22.2, mlx-rs commit
`adf439bf5361d8a02955de7aaf651b0c45900945` and native MLX v0.32.2.
Online documentation can move; pinned implementations were inspected where
capability/dispatch claims matter.

- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html),
  [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html),
  [Metal debugging](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html).
- [MLX v0.32.2 fast operators](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/fast.cpp),
  [Metal attention dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp).
- [Pinned mlx-rs compilation contract](https://github.com/oxiglade/mlx-rs/blob/adf439bf5361d8a02955de7aaf651b0c45900945/mlx-rs/src/transforms/compile/mod.rs).
- [CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
  [best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html),
  [memory pools](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html).
- [Tokenizers 0.22.2 tokenizer APIs](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/tokenizer/mod.rs),
  [BPE word implementation](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/bpe/word.rs),
  [WordPiece decoder](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/decoders/wordpiece.rs).
- [Safetensors](https://huggingface.co/docs/safetensors/index),
  [Rust Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html).

[Source retrieval metadata](results/2026-11-11-phase2-optimization-review/sources.json)
records URLs/hashes, including an unsuccessful URL and its corrected lookup.
[Derived counts](results/2026-11-11-phase2-optimization-review/derived.json)
record arithmetic and input hashes. Downloaded third-party source/docs remain
under ignored `.reference/phase2-next-optimization-research/`; no weights or large
traces were added. No new speedup, GPU correctness or energy result is claimed.

Review validation: `CARGO_BUILD_JOBS=4 ./tools/check.sh` passed, including 24
report-tool tests. Research JSON, recorded input/download hashes, static count
arithmetic and local documentation links were checked; `git diff --check` and
an untracked-report whitespace check passed. No CUDA/Metal rerun was performed
for this documentation-only review. The check log is local at
`.reference/phase2-next-optimization-research/check.log`.
