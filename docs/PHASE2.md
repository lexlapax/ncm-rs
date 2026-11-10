# Phase 2 contract and acceptance tracker

Status: **reopened at user request for performance research and remediation**.
The original Linux/CUDA and Apple/MLX correctness/measurement baselines are
published; their negative performance findings now block advancement to Phase 3.
See [the extension plan](PHASE2-PERFORMANCE.md) for ordered work and new gates.
Phase 1 evidence remains unchanged; this document declares Phase 2 criteria before
Rust codec implementation. The confirmed scope includes Sudoku codec execution
and TinyStories/GPT-2/Qwen/BERT tokenizer asset conversion/parity, not their full
text models. Linux/CUDA acceptance must precede Apple/MLX implementation/testing.

## Reference identity and supported initial model

NCM `c70274273f489abc73832b2b98584a8483b6b203`, HF checkpoints
`d342ce8aaea58830eef3b8ad473664673e41f692`, Torch 2.15.0. The inspected canonical
Sudoku config/codec/training file hashes are anchored in the committed Phase 1
manifest, not accepted from an untrusted replacement manifest. Model tests may use
a separately identified tiny synthetic codec, never mislabel it as trained weights.

Sudoku-easy codec resolution from `models/vae.py`, `vqvae.py`, `quant.py`:

- Input vocabulary 12, length 89, latent width 256. `n_layers=4` means **four
  attention heads**, not four decoder blocks. Encoder has token + learned position
  embeddings, no blocks/conv stack/compression. Dropout is disabled in eval mode.
- Pre-quant GroupNorm: eight groups, no affine terms, epsilon 1e-5; reduction spans
  sequence and grouped channels. Encoder and decoder compression factor is one.
- Levels `[1,2,4,6,9,13,18,26,37,50,67,89]`, each with 16 codes; concatenated length
  322. Level zero learned cascade factors 89 into a single stride/kernel-89 conv
  and transposed conv; the base-four setting does **not** make this a stride-four
  operation. Cascade normalization is off in this checkpoint.
- Nonterminal later levels use adaptive-area downsampling and half-pixel linear
  upsampling. The final level uses the full residual/codebook values directly.
- Distances use `sum(x*x) + sum(e*e) - 2*(x @ e.T)`, not normalized cosine distance.
  Argmin chooses the first minimum. Each contribution is refined by its own
  kernel-three Phi conv, with `(1-r)*input + r*conv(input)`, `r=0.5`.
- Accumulate the refined contribution and subtract it from the residual at each
  level. Expose independent cumulative snapshots; never alias a mutable accumulator.
- Decoder has one sequential pre-LayerNorm residual block, full **bidirectional**
  attention, Q/K RMSNorm, an exact-erf GELU 4× MLP, final weight-only LayerNorm,
  and bias-free 256→12 output projection. LayerNorm/RMSNorm epsilon is 1e-6.
  No decoder positional embedding. `models/common.py` defaults to causal attention
  without a mask; it is not interchangeable with the codec's `models/vae.py`.

The 58 runtime tensors are: 2 encoder embeddings; 16 decoder parameters; 12
codebooks; 4 learned down/up weight/bias tensors; and 24 Phi weight/bias tensors.
Full names/shapes/dtypes/hashes remain in the Phase 1 artifact manifest. All are
F32; the 36 EMA/count/hit buffers are retained separately and are not inference
weights. Nonpersistent `_fd_step` is training-only. Reject missing/extra or wrongly
shaped runtime weights, unknown inference-critical fields, invalid token/code IDs,
unsupported model options and input lengths before device work.

## Predeclared numerical policy: codec-fp32-v1

Elementwise `abs(actual-reference) <= atol + rtol*abs(reference)`, with shape/dtype
checks and rejection of unexpected nonfinite values. Fixed masks may contain -inf
only in explicitly declared mask positions. Existing Phase 1 tolerances stay fixed.

| Boundary | atol | rtol |
| --- | ---: | ---: |
| Existing and new isolated primitives / stable selection fixtures | 1e-5 | 1e-4 |
| Encoder, pre-quant, single-level input/upsample/Phi and decoder submodules | 1e-4 | 2e-4 |
| Squared codebook distances (256-coordinate reduction/cancellation) | 1e-3 | 2e-4 |
| Twelve-level cumulative reconstructions, full decoder block and final logits | 5e-4 | 2e-4 |

These FP32 accumulation budgets are declared before candidate results, not relaxed
in response to mismatches. Capture max/mean absolute error, relative error with
zero-reference handling, mismatch counts and IDs. New justified tolerance changes
require explicit review; rewriting a golden to hide a failure is prohibited.

For a reference score row, call a winner stable when its top-two gap exceeds
`2 * (atol + rtol * max(abs(row)))` using that row's declared score budget. Require
exact IDs for every stable winner and for synthetic known-margin argmin/argmax
fixtures; separately test first-index tie semantics. Report **all** checkpoint
code/output disagreements, minimum margins and unstable counts. Near-tie labels
never waive floating-tensor failures or permit dropping difficult samples.
Free-running encode/decode and teacher-forced fixed-code decode are both required;
measure reconstruction agreement against Python, not perfect recovery of inputs.

## Fixture and safety contract

Canonical oracles are CPU for the tiny synthetic fixture and CUDA for the trained
Sudoku checkpoint; both GPU adapters must use the same respective reference.
The optional full-checkpoint CPU capture is a cross-device diagnostic, not a third
qualified runtime. The initial audit found one out-of-budget CPU/CUDA internal
learned-down projection value on the blank board (a 22,784-term projection with
cancellation); selected IDs and final reconstructions agree. Keep that failure in
the report. **No tolerance is changed** and it does not exempt a future CUDA/MLX
candidate from the declared trained-CUDA-reference gate.

- Preserve all Phase 1 fixtures. New fixture IDs/schema identify all new content.
- Tiny CPU oracle: deterministic untrained width-16 codec, four heads, length 16,
  levels `[1,3,7,16]`, four-entry codebooks, two-stage `[4,4]` learned cascade with
  intermediate/final GroupNorm. This exercises functionality absent in the single-
  stage Sudoku checkpoint. Synthetic parameters may be small committed fixtures;
  downloaded trained weights and large traces remain outside Git. The tiny fixture
  interns duplicate arrays and stores typed little-endian hex bytes, preserving
  exact FP32/I64 bits without huge decimal dumps. Integer reference IDs may use
  a backend's native index dtype, but conversion must be checked and IDs exact.
- Real checkpoint: deterministic solved, blank and partially blank synthetic boards,
  batch-one and batch-two cases, plus fixed supplied codes. No official evaluation
  data/pickle or claim about Sudoku solving is involved.
- Capture encoder/pre-quant, per-level distances/IDs/contributions/residuals and
  cumulative snapshots, decoder submodule inputs/outputs, final logits/IDs and
  selection margins. Instrumentation clones observations without changing upstream
  math. Decode snapshots are shadow sums of the actual contributions, checked
  against the unmodified `idx_to_z_q` result; do not use aliased `idx_to_zhats` or
  the broken `reconstruct` helper.
- Pin and hash any new tokenizer asset before conversion. TinyStories byte-valued
  vocabulary/merge data must remain byte-safe; no implicit unrestricted pickle.
  HF wrappers must preserve special tokens, postprocessors and cleanup behavior.
  Exact asset revisions/hashes and conversion rejection tests are required before
  that separate task can pass. No multi-GB text checkpoint download is necessary.

## Runtime / measurement requirements

Use a minimal shared Rust codec over backend operations, immutable weights and
request-local state. Explicitly evaluate MLX graphs and synchronize CUDA timings.
Do not impose unsafe Send/Sync on MLX arrays. A tiny CPU path is a testing oracle,
not a third performance product. Typed runtime config/loading is distinct from the
Phase 1 inspection loader's integrity-only checks.

Benchmark uninstrumented codec paths separately from trace capture: encode,
fixed-code decode, reconstruction, first execution, warm execution and load.
Use the reporting protocol (three fresh processes, five warmups, thirty trials,
batch-one latency and small matching batches), exact workload/source hashes and
host/native allocator memory observations. Profile the known Apple load/RSS issue
before any efficiency claim. Preserve negative findings; no speculative custom
kernels, compiled mixed-precision policy or generation-cache work in this phase.

## Original baseline acceptance (historical, still valid)

The completed items below establish correctness and measurement, not satisfaction
of the newly requested performance-remediation gate.

- [x] Reference contract, fixtures and supported tensor/config inventory: see the
  [oracle audit](results/2026-11-10-phase2-oracle/README.md). This is not Rust parity.
- [x] Validated shared configuration/backend/model-loading interface: [construction
  checks](results/2026-11-10-phase2-runtime/README.md). Construction/readback alone
  is not forward parity; the subsequent full codec check is recorded below.
- [x] Linux CPU/CUDA primitive/component parity: [component report](results/2026-11-10-phase2-primitives/README.md).
- [x] Complete Linux codec parity: [full tiny/trained comparison](results/2026-11-10-phase2-codec/README.md),
  including free-running and fixed-code paths, near ties and uninstrumented outputs.
- [x] Sudoku regression plus TinyStories/HF tokenizer conversion/parity on Linux:
  [asset pins, safety, exact cases](results/2026-11-10-phase2-tokenizers/README.md).
  Apple must repeat the same fixtures after Linux platform acceptance.
- [x] Published Linux reconstruction, performance and memory gate:
  [three-process codec/tokenizer evidence](results/2026-11-10-phase2-linux/README.md).
  Native codec is slower than eager Torch; measured memory scopes differ. No
  efficiency claim or compiled-Python qualification is implied.
- [x] Corresponding Apple/MLX implementation, parity and measurements:
  [codec/tokenizer parity](results/2026-11-10-phase2-mlx-codec/README.md) and
  [three-process runtime evidence](results/2026-11-10-phase2-mlx/README.md) pass.
  The adapted Python MLX baseline is independently qualified against the unchanged
  CUDA oracle; native host RSS remains a disclosed negative finding.
- [x] Both local checks, reports, context updates and reviewed commit/push closure:
  acceptance evidence is published in `7a74fb3`. Large licensed trace/tokenizer
  archives remain separate release assets pending explicit publication authorization.

## Extended acceptance (required before Phase 3)

- [x] Audit fair scopes, synchronization and clean-checkout report reproduction;
  refresh paired baselines and declare noise-aware thresholds before optimization.
- [x] Complete source/documentation/upstream research and causal profiling on Linux.
- [x] Fix Linux codec/loading/tokenizer deficits; retain unchanged correctness and
  publish candidate versus previous Rust versus Python performance/memory evidence.
- [ ] After Linux acceptance, research and fix Apple load time/host-memory
  amplification and tokenizer deficits; preserve execution performance and parity.
- [ ] Accept both extended gates and publish final evidence and updated status.

The [CUDA remediation milestone](results/2026-11-10-phase2-cuda-remediation/README.md)
qualifies vectorized/fused/view/pool changes and publishes three-way measurements.
[Extended Linux acceptance](results/2026-11-10-phase2-linux-acceptance/README.md)
now passes, including uncertainty controls, lifecycle evidence and a whole-matrix
audit. Apple remediation is next; Phase 2 as a whole is not yet accepted.

The [extension plan](PHASE2-PERFORMANCE.md) defines predeclared non-inferiority bands,
source-cited research deliverables and blocking/exception rules. Historical
slowness/high RSS is no longer accepted solely because it was disclosed. Narrow,
profile-justified FP32 backend kernels may be investigated; speculative kernels,
precision changes, weaker validation and generator caching remain out of scope.

Missing hardware/artifact evidence remains pending. If progress requires changed
compatibility requirements, unavailable resources or unsafe conversion, stop and
ask the user rather than silently narrowing scope. Phase 3 remains out of scope.
