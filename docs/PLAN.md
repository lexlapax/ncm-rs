# Phased port and validation plan

The user approved implementation on 2026-11-09. Phases 0 and 1 are complete.
**Phase 2 is reopened for performance research and remediation at the user's
request. Phase 3 is blocked.** Its original correctness/measurement baselines were
published in Linux-then-Apple order, but merely documenting slower native execution
or excessive memory no longer satisfies the extended acceptance gate. **Within
every phase, implement and validate Linux/CUDA first, then macOS/MLX.** Complete
both platform gates before advancing to the next phase; do not finish the whole
CUDA project before starting Apple work. GPU feasibility and artifact work precede
committing to a detailed delivery estimate.

## Phase 0 — research, structure, and review (complete)

Deliver the workspace, provenance, architecture proposal, compatibility inventory,
artifact/test contracts, upstream maintenance policy, reporting requirements and
CPU scaffold CI. No inference implementation or performance claims.

**Gate:** formatting, Clippy, workspace tests/build and docs succeed; tests currently
contain no numerical cases. Commit/push scaffold; obtain review before Phase 1.

## Phase 1 — reference contract, artifacts, and runtime spikes (complete)

Both platform gates passed in the required Linux-then-Mac order on 2026-11-10.
See the [acceptance checklist](PHASE1.md) for evidence, scope and negative findings.
The scoped Sudoku prototype round-trips all 238 tensors exactly on CPU/CUDA/MLX;
56 original-vs-converted upstream CUDA traces and tokenizer goldens are retained.
All 18 operator cases pass on all three devices. Load/memory and three-process
baselines are published, including slower resampling and poor Apple load/RSS results.
This is not a general runtime importer, Rust model parity or inference release.
Build/test checks run locally via `tools/check.sh` with explicit `--cuda` / `--mlx`;
the metadata-only upstream watcher remains the only GitHub workflow.

Original phase tasks (acceptance scope/dispositions are detailed in the checklist):

1. Pin Python NCM, Torch/dependencies, HF checkpoint/data/tokenizer revisions and
   checksums. Start with Sudoku easy, then TinyStories. Download only needed files.
2. Implement an offline exporter: weights-only loading first; explicit isolated
   handling of trusted legacy pickle; safetensors + versioned JSON, including
   inference unigrams, aliases, special-token settings and provenance.
3. Inventory every tensor name, shape, dtype and persistent/nonpersistent buffer;
   separate inference weights from EMA/training state. Validate canonical reload.
4. Build a small fixture generator around the pinned dense/uncached Python path.
   Export layer inputs/outputs, per-level codes/logits/reconstructions and final IDs.
   Record reference patches and RNG inputs; do not silently fix the oracle.
5. Spike **both** adapters on real devices, Linux/CUDA first and macOS/MLX second,
   before freezing the backend interface:
   matmul, masked noncausal attention, conv/transposed conv, GroupNorm, exact
   resampling, dtype conversions, synchronization, loading and memory behavior.
6. Establish a small real-device microbenchmark harness with machine-readable
   results. Record baseline Python eager timings; reproduce `torch.compile` where
   feasible and label it separately. Establish upstream update/report workflows.

**Gate:** canonical weights round-trip without numerical changes; tokenizer IDs
match; tiny operator fixtures execute on CPU, CUDA and MLX. Publish capability and
baseline report, including unsupported operations. MLX needs Apple hardware access;
a Linux-only result does not pass this gate. No broad framework written in advance.

## Phase 2 — shared primitives, codec and performance remediation (reopened)

The [Phase 2 contract](PHASE2.md) fixes the initial model scope, tensor inventory,
new numerical budgets and fixture semantics. Tiny CPU and trained CUDA reference
captures, passing Rust CUDA/MLX codec trace comparisons and pinned tokenizer parity
are published. The [Linux quality/performance/memory gate](results/2026-11-10-phase2-linux/README.md)
and [Apple quality/performance/memory gate](results/2026-11-10-phase2-mlx/README.md)
established the original measured baseline. Linux native codec latency is roughly
4–5× slower than eager Torch; Apple checked loading is about 3.25× slower with
roughly 12–15× higher process host HWM than its independently qualified adapted
Python MLX baseline. These deficits now require investigation **and remediation**,
not merely disclosure. Historical reports remain valid but do not close this
extension; see the [detailed research and remediation plan](PHASE2-PERFORMANCE.md).

Implement validated configs, backend operations and deterministic fixture harness.
Port norms, split-half RoPE, bidirectional/prefix attention, GELU blocks,
convolution/cascade samplers, exact area/linear interpolation, quantizer codebooks
and residual accumulation. Then codec encoder, pre-quant normalization, decoder,
encode/decode and explicit per-level snapshots.

The original FP32 correctness-first implementation and Sudoku/HF/TinyStories
asset conversion are complete. Preserve these semantics and the tiny CPU oracle.
The extension permits narrowly justified backend optimization, including audited
kernels only when source research and profiling demonstrate a need; speculative
kernels, reduced precision and generator/prefix caching remain out of scope.

[CUDA remediation evidence](results/2026-11-10-phase2-cuda-remediation/README.md)
now includes qualified production changes and three-way measurements: all codec
latency cells meet the predeclared Python bands. [Extended Linux acceptance](results/2026-11-10-phase2-linux-acceptance/README.md)
now passes uncertainty/audit/startup follow-ups. Apple remediation is next;
Phase 3 remains blocked.

### Required extension, in order

1. Audit benchmark fairness, synchronization, retained outputs, startup/load/drop
   scope and clean-checkout reproducibility. Refresh old Rust and Python baselines,
   measure process-level noise and lock acceptance thresholds before changing code.
2. Research pinned NCM/PyTorch/Candle implementations, native CUDA APIs and relevant
   upstream findings. Profile CPU submission, kernel launches, layout copies,
   reductions, allocations and I/O. Produce source-cited, testable causal findings.
3. Fix Linux/CUDA deficits first: per-position resampling compositions are an
   initial hypothesis, not the only target. Also investigate decoder scaling,
   selection, shared checked loading and tokenizer loading. Qualify every change
   against unchanged outputs and publish old-Rust/new-Rust/Python comparisons.
4. Only after extended Linux acceptance, research/profile Apple MLX ownership,
   host allocations and allocator retention. Fix slow/high-RSS codec loading and
   tokenizer load deficits without losing the existing execution advantage.
   Any further shared fix returns through Linux validation before Apple testing.
5. Publish both extended gates and update acceptance before Phase 3. Require
   comparable latency and memory versus Python within predeclared uncertainty,
   not just improvement over old Rust; preserve existing native advantages.
   Predeclared maximum bands and blocking/exception rules are in the detailed plan.

**Gate:** operator, block, encode/decode and reconstruction fixtures satisfy
predeclared tolerances on both GPUs; exact stable argmin/argmax fixtures and layout
checks pass. Report reconstruction agreement versus Python, not an unrealistic
claim that this lossy codec recovers all arbitrary text perfectly. Additionally,
all extended performance/memory criteria in [PHASE2-PERFORMANCE.md](PHASE2-PERFORMANCE.md)
must pass on both platforms. Explanations without fixes do not suffice; remaining
material deficits block closure unless the user explicitly accepts an exception.

## Phase 3 — end-to-end Sudoku vertical slice

**Not started; blocked on the extended Phase 2 performance/memory gates.**

Implement shared generator, raw prefix path, learned level-zero queries, level
embeddings/RoPE, untied heads, eager level loop, greedy selection and final codec
decode. Add a real CLI for artifact inspection, reconstruction and generation;
batched input/output, bounds checks, explicit device selection and useful errors.
Keep weights immutable and all per-request state local. Do not enable prefix caching.

**Gate:** identical stable golden codes/final IDs on both runtimes, then compare
all three Sudoku checkpoints on the specified 2,000-board evaluations. Report
solved, cell, valid-grid and clue-preservation rates, malformed output counts,
latency, throughput and peak memory. Publish the **first end-to-end comparison**
against pinned Python and retain raw per-board outputs. No silent CPU fallback.

## Phase 4 — released text-profile compatibility

Port incrementally, running the full preceding regression suite after each profile:

1. **TinyStories:** custom byte BPE, learned first-level cascade, codebook-distance
   heads, supplied unigram initial codes, stochastic sampling.
2. **XSum:** GPT-2 special tokens, multi-position level-zero queries, long padded
   prefixes, cascade GroupNorm, frequency/deficit/adjacency decoding penalties, CFG.
3. **WMT14:** Qwen tokenizer, prefix-derived level zero, untied bias-free heads,
   CFG and the exact source padding/truncation policy.
4. **LM1B:** BERT tokenizer/postprocessing, initial unigram override of query mode,
   shared output head with per-level biases, post-upsample conv and four-block
   positional codec decoder; small first, then medium/large with memory budgets.
5. **OWT:** synthetic/config/operator coverage now; checkpoint and quality coverage
   only when weights are actually released and validated. Preserve no-EOS-cut policy.

Defer unused research options (T5 context encoder, alternate block layouts, GQA,
SwiGLU/GEGLU/ReLU², autoregressive level zero, learned-all sampling, compression,
trainable tied heads) until requested or used by a tracked upstream release.
Reject unsupported inference configurations rather than approximate them silently.

**Gate:** each profile gets tokenizer parity, teacher-forced numerical parity,
end-to-end output checks and a published quality/performance report on both GPUs.
Use official protocols for BLEU/chrF, ROUGE, generative PPL and entropy. Verify
checkpoint metadata before advertising support. A blocked artifact stays blocked.

## Phase 5 — optimization and inference release

Profile real batch/level shapes, then prioritize demonstrated bottlenecks:

- Correct reusable prefix K/V, including padded prefixes, level-zero queries and
  conditional/null CFG branches; never cache old target levels as token history.
- Device-resident reconstruction/sampling, bounded lazy graphs, allocation reuse,
  precomputed norms/positions/resampling maps and layout preparation.
- Optimized masked/noncausal attention, fused norm/QKV and interpolation kernels
  only when they beat tested compositions. No unconditional FlashAttention assumption.
- MLX compilation for stable shapes; CUDA graph experiments for launch-bound
  paths; record compile/capture costs separately.
- BF16/FP16 policy and optional quantization with separate quality gates. FP32
  compatibility remains available. Quantized codec/codebooks are not the first target.
- Streaming bounded batches, cancellation/OOM recovery, multi-request isolation,
  release packaging and reproducible environment documentation.

**Gate:** full regression/quality suite passes, cache-on/off equivalence passes,
no unreviewed quality regression, and published reproducible before/after reports
show actual gains on tested workloads. Revert optimizations without useful measured
benefit. Set numeric latency budgets after Phase 1 baselines, not invented targets.
Publish a tagged inference release with all supported profiles and limitations.

## Phase 6 — training port/reconstruction (separate approval gate)

This is not a mechanical translation: upstream lacks the full trainer. Request
missing recipes and track upstream changes before filling gaps independently.
Label any reconstructed recipe distinctly from an upstream reproduction.

1. Verify autodiff coverage on both runtimes, including conv/transposed conv,
   custom ops, masking, gather and codebook detaches; finite-difference tests.
2. Frozen-codec generator training first: teacher-forced all-level inputs,
   block-diagonal masks, clean labels, conditioning corruption, level-weighted
   cross-entropy, AdamW, schedules, gradient clipping and resume checkpoints.
3. Codec training: reconstruction/VQ losses, straight-through estimator, EMA
   counts/sums, nested dropout, code restarts and optional k-means initialization.
   Resolve paper/code discrepancies and gradient-flow details with explicit tests.
4. Single-device tiny-overfit then small-data experiments on both GPUs; compare
   losses, gradients, EMA updates, optimizer steps and resume determinism with
   a pinned Python reference. Only then pursue distributed CUDA/NCCL training;
   Apple single-device training is the first Apple target.

**Gate:** finite-difference/gradient/EMA tests, tiny overfit, checkpoint resume and
budget-matched learning curves. Publish quality AND compute/memory cost, including
negative results. Do not claim paper-scale reproduction from toy overfit.

## Continuous lane — upstream parity and published comparisons

Runs alongside every phase, not merely after the first release:

- Weekly automated upstream/code/checkpoint inventory check; monthly human triage;
  expedite changes affecting supported releases. Track old supported and candidate
  references separately. See [UPSTREAM.md](UPSTREAM.md).
- Every execution/output-affecting change carries fresh before/after comparisons
  for affected workloads and hardware; releases/upstream upgrades run the complete
  support matrix. Missing hardware is a visible pending gate, not a pass.
- Publish readable summaries plus machine-readable results with exact Git IDs,
  environments, artifacts, evaluator versions and raw-output digests. See
  [REPORTING.md](REPORTING.md). No benchmark numbers in this scaffold are fabricated.

## Approval decisions and external requirements

- Inference-first plan approved. The user requires Linux/CUDA then macOS/MLX
  within each phase. Phase 1 may revise runtime details with evidence, without
  silently replacing MLX with generic Metal.
- Apple runner authorized via `ssh macstudio`: M1 Ultra, 64 GiB, Xcode/Metal and
  Homebrew Rust available (prepend `/opt/homebrew/bin` to noninteractive PATH).
  The local RTX 3060 Ti is useful for correctness/small batches; large models and
  paper-scale evaluation may require a larger CUDA GPU.
- Repository is public at the user's request. Publish reviewed comparison summaries
  in `docs/results/` and durable raw bundles as release assets; a Pages site can
  be added later without blocking report publication.
- Python remains a development-only dependency for trusted conversion/reference
  evaluation. Remove it from deployed inference, not from the scientific oracle.
