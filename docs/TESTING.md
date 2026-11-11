# Test strategy and acceptance gates

Phase 1 is complete: 18 fixed operator cases pass CPU/CUDA/MLX, all 238 checkpoint
tensors round-trip/read back exactly, and the tokenizer, artifact and real-Metal
tests exist. There are 29 report-tool tests. Phase 2's unchanged
[codec reference contract](PHASE2.md), tiny synthetic fixture and trained CUDA
oracle now qualify both GPU adapters. [Final dual-platform acceptance](results/2026-11-10-phase2-acceptance/README.md)
passes 642 trained traces / 2,500 exact decisions per GPU, with two near ties
disclosed; pinned tokenizer goldens/rejections and all 78 performance/memory
comparisons per platform also pass. There is no official generation/task-quality
result. The [oracle audit](results/2026-11-10-phase2-oracle/README.md) retains its
separate original-Python CPU/CUDA diagnostic failure. Those results qualify the
earlier accepted candidate. Phase 2 is reopened for the
[additional five-step plan](PHASE2-OPTIMIZATION-REVIEW.md); it requires fresh
Linux-first/Apple-second gates against current accepted Rust and qualified Python,
plus CPU/allocation and lifecycle evidence. New runtime checks remain pending;
a default documentation check does not satisfy them. The remaining model gates
below are still planned.

## Layers of evidence

| Layer | Cases | Runs |
| --- | --- | --- |
| Config/artifact | Bounds, unknown fields, invalid head counts, bad IDs, sizes, hashes, truncated files, path traversal, missing metadata | CPU, every PR |
| Text tokenization | Unicode/bytes, whitespace, special tokens, cleanup, exact encode/decode, BERT/Qwen postprocessors, Sudoku separators | CPU, every relevant PR |
| Tensor semantics | Norms, RoPE, conv/transposed conv, interpolation, masks, heads, sampling | Tiny CPU fixtures every PR; both GPUs before runtime approval |
| Model components | Encoder, quantizer, cumulative snapshots, decoder, prefix encoding, each generator block/level | Deterministic golden inputs on CPU and both GPUs |
| Generation | All level-zero modes, heads, EOS policies, penalties, CFG, batch sizes, request isolation | Both GPUs for each supported profile |
| Quality | Reconstruction, Sudoku, BLEU/chrF, ROUGE, PPL/entropy | Smoke on relevant changes; full on releases/upstream upgrades |
| Performance | Cold load/first execution, warm generation, memory, per-level breakdown, throughput | Dedicated real-device runners; reports per relevant change |
| Training (later) | Gradients, EMA, optimizer updates, tiny overfit, resume, learning curves | Separate Phase 6 matrix |

Golden fixtures contain source pins, normalized config, tensor checksums,
inputs/outputs, intermediate tensors, dtype/device/math policy, and oracle patches.
Small synthetic fixtures may be committed after licensing/size review; downloaded
weights/datasets and large traces stay outside Git. Regeneration is reviewed, not
a routine way to make failing tests pass.

## Numerical comparison policy

Use `abs(actual-reference) <= atol + rtol*abs(reference)` elementwise, with separate
finite-value and mask-position checks. Initial operator targets are FP32
`atol=1e-5, rtol=1e-4`; initial FP16/BF16 exploratory targets are `atol=1e-2,
rtol=1e-2`. These are **starting targets**, not blanket end-to-end tolerances.
Calibrate per operation on pinned fixtures, document error distributions, and
approve tolerances before optimization; never silently widen them to fit failures.
Report max/mean absolute and relative error, and disagreement rates. End-to-end
accumulation may require a distinct justified budget.

- Floating logits alone are insufficient: compare predicted code IDs, latent
  reconstructions, final token IDs and quality metrics.
- Require exact IDs on stable-margin deterministic fixtures. Diagnose near-tie
  argmin/argmax flips using the winning margin and mathematical reference; report
  real-model disagreements explicitly instead of excluding difficult samples.
- Baseline FP32 disables approximate math such as TF32 where configurable; record
  settings. Test each mixed-precision policy separately, with FP32 reductions where
  needed. Verify default dtype promotion instead of assuming it matches Torch.
- Masks include negative infinity intentionally; compare allowed/disallowed
  positions separately. Fail unexpected NaNs/infinities. Explicitly reject or
  define all-masked prefix behavior and test empty/truncated prompts.
- LayerNorm/RMSNorm/GroupNorm have different epsilons/axes; test constant inputs,
  extreme values, noncontiguous inputs, and groups containing multiple positions.
- RoPE tests use unequal prefix/target lengths and multiple levels, with reset
  positions and split-half rotation. Test cached keys are not rotated twice.

### Resampling and convolution traps

For adaptive-area output bin i, test source indices from
`floor(i * input_len / output_len)` to
`ceil((i+1) * input_len / output_len)` (end-exclusive), including overlapping bins.
For linear interpolation, test half-pixel mapping and clamped endpoints with
`align_corners=False`. Include length 1, equal lengths, uneven and prime lengths,
non-power-of-two ladders and the 89-stride learned Sudoku first-level sampler.
Test ConvTranspose1d independently using asymmetric weights/impulses; a simple
square identity test can hide a weight-axis bug.

### Sampling and decoding

- Greedy, top-k disabled/boundary, tied kth values, top-p 0/1/boundaries,
  temperature validation, all-invalid logits, variable vocab sizes.
- Feed explicit uniforms into a reference categorical sampler where possible;
  also force upstream codes at each level for local numerical comparisons.
  Native Torch and MLX/CUDA RNG streams need not match from equal seeds.
- Distributional tests use fixed sample counts and predefined statistical tests;
  a single matching stochastic output is not proof of correctness.
- Initial unigram selection is separate from generator sampling. Preserve LM1B's
  supplied-code override, even though the architecture supports query level zero.
- Test CFG scale=0 against no null branch; preserve guidance ramp, null EOS/PAD,
  batch ordering and identical sampled codes for paired branches.
- Compare final sequential frequency/deficit/adjacency selection using identical
  decoder logits and FP64 CPU reference, including repeated tokens and ties.
- Match EOS trimming and special-token removal; OWT does not cut at the first EOS.

### Stateful and cache checks

Repeat requests in different orders/batch sizes, cancellation/error followed by
reuse, and concurrent worker isolation. Test memory returns to a bounded steady
state. Correct prefix caching must agree with dense uncached inference for every
supported prefix/level-zero/head/CFG combination; padding must never become visible.
Per-level snapshots must own immutable values rather than alias the final accumulator.

## Quality protocols

- **Codec:** exact token reconstruction and per-level/codebook statistics on the
  same held-out inputs as Python. Do not demand 100% recovery of arbitrary input.
- **Sudoku:** all 2,000 test boards per difficulty; exact solution, cell accuracy,
  valid rows/columns/boxes, clue preservation, and malformed lengths. Publish both
  faithful upstream metrics and stricter diagnostics (upstream zipped-cell scoring
  can obscure malformed output lengths).
- **WMT14:** 3,003 examples, cased/lowercase sacreBLEU with full signature, chrF,
  length ratio. Base and CFG runs separately.
- **XSum:** 11,334 examples; stemmed ROUGE-1/2/L F1 using the same aggregation;
  generative PPL oracle matching the evaluation script. The script uses GPT-2 here,
  unlike unconditional GPT-2-large; pin the actual evaluator implementation.
- **TinyStories/LM1B/OWT when available:** five seeds `[42,0,1,2,3]`, 256 samples
  each, generator config sampling and the exact protocol. LM1B FLM scoring preserves
  specials, rescores the first 128 GPT-2 tokens and aggregates corpus NLL; it is
  not interchangeable with mean per-sequence perplexity. Retain raw and retokenized
  IDs/entropy definitions. Report seed variation and confidence intervals.

Use the same frozen external evaluator for Python and Rust outputs initially;
it is not a deployed inference dependency. Later Rust metric ports must match
that oracle (including stemming, tokenization and PPL aggregation). Store outputs
so scoring can be rerun without regeneration. Define non-inferiority margins on a
fixed validation set before running held-out comparisons; deterministic compatible
profiles should match, stochastic profiles need uncertainty-aware comparisons.

## Local validation and hardware policy

At the user's request, GitHub build CI is removed. Run `./tools/check.sh` before
committing: fmt, default build check, CPU-feature Clippy/tests/docs, and Python
report-tool tests/syntax checks. This needs Rust 1.99.1+, rustfmt, Clippy and Python
3, but no Torch or GPU SDK. ShellCheck runs too when installed. The old Rust 1.88
support claim failed on Candle's x86 intrinsics; 1.99.1 is the tested conservative
floor. Supporting older toolchains requires explicit local qualification.

Opt-in `./tools/check.sh --cuda` additionally lints the CUDA feature and runs the
real-device probe in release mode, refusing a missing CUDA feature/device or a
non-Linux platform rather than falling back. Use the documented host compiler and
compute capability environment variables. It writes one smoke report, not the
three-process Python/Rust comparison required for performance changes.

Opt-in `./tools/check.sh --mlx` adds release MLX Clippy, real-Metal tests, docs and
the synchronized operator probe. It rejects non-Apple-Silicon targets. Configure
the native MLX prefix/rpath using the [reference instructions](../tools/reference/README.md).
This check never installs MLX, downloads weights or launches SSH automatically.

Only the metadata-only scheduled upstream watcher remains on GitHub. Do not
reintroduce build CI without user approval. Add future model suites to
the local workflow as implemented. Full benchmark/quality reports are still
required for releases and relevant changes, independent of where checks execute.
Use separate platform feature selections, not cross-platform `--all-features`.

Within every phase, Linux/CUDA completes its checks before macOS/MLX starts.
Phase 1 completed in that order; see the [acceptance checklist](PHASE1.md).
Recheck `macstudio` availability before later hardware runs and inspect its checkout
before pulling.
Never run untrusted contribution code on persistent local/remote GPU machines or
expose credentials to it. Only reviewed results become baseline/public reports.
See [REPORTING.md](REPORTING.md) for publication and regression policy.
