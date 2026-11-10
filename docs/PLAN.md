# Phased port and validation plan

The user approved implementation on 2026-11-09. Phase 0 is complete; Phase 1
is in progress. **Within every phase, implement and validate Linux/CUDA first,
then macOS/MLX.** Complete both platform gates before advancing to the next phase;
do not finish the whole CUDA project before starting Apple work. GPU feasibility
and artifact work precede committing to a detailed delivery estimate.

## Phase 0 — research, structure, and review (complete)

Deliver the workspace, provenance, architecture proposal, compatibility inventory,
artifact/test contracts, upstream maintenance policy, reporting requirements and
CPU scaffold CI. No inference implementation or performance claims.

**Gate:** formatting, Clippy, workspace tests/build and docs succeed; tests currently
contain no numerical cases. Commit/push scaffold; obtain review before Phase 1.

## Phase 1 — reference contract, artifacts, and runtime spikes (in progress)

Current milestone: Linux CPU/CUDA candidate operators pass 18 synthetic PyTorch
fixtures, with a [public microprobe report](results/2026-11-09-cuda-operators/README.md).
The scoped Sudoku artifact milestone now has 238 byte-exact tensor roundtrips,
CPU/CUDA readback, tokenizer goldens and 56 upstream Python model traces unchanged
by conversion ([report](results/2026-11-10-sudoku-artifacts/README.md)). This is not
a general runtime importer or Rust model parity. Load-performance/memory evidence
and the remaining runtime-contract checks are still open.
The Apple pass has not started. The user has now confirmed `macstudio` is available;
announce MLX validation once Linux checks are complete (recheck availability in
later sessions). No gate has been waived by the operator-only evidence.
Build/test checks now run locally via `tools/check.sh`, not GitHub build CI;
the weekly upstream metadata watcher remains enabled.

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

## Phase 2 — shared primitives and codec

Implement validated configs, backend operations and deterministic fixture harness.
Port norms, split-half RoPE, bidirectional/prefix attention, GELU blocks,
convolution/cascade samplers, exact area/linear interpolation, quantizer codebooks
and residual accumulation. Then codec encoder, pre-quant normalization, decoder,
encode/decode and explicit per-level snapshots.

Start FP32, with simple composed ops; add no speculative custom kernels.
Implement Sudoku tokenization and HF/TinyStories asset conversion in parallel.
Use a tiny CPU oracle for continuous integration, not a high-performance CPU product.

**Gate:** operator, block, encode/decode and reconstruction fixtures satisfy
predeclared tolerances on both GPUs; exact stable argmin/argmax fixtures and layout
checks pass. Report reconstruction agreement versus Python, not an unrealistic
claim that this lossy codec recovers all arbitrary text perfectly.

## Phase 3 — end-to-end Sudoku vertical slice

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
