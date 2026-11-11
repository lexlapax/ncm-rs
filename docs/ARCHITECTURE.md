# Architecture proposal

Status: [Phase 2 codec/tokenizer acceptance passes](results/2026-11-10-phase2-acceptance/README.md)
on Linux/CUDA then macOS/MLX. The shared codec/backend contract is implemented and
validated for that scope; generator, sampling and inference-CLI architecture below
remains planned. Crate boundaries do not imply full-model APIs or frozen abstractions.
Phase 2 is reopened for the [approved additional optimization plan](PHASE2-OPTIMIZATION-REVIEW.md);
[lower-risk Linux experiments](results/2026-11-11-phase2-low-risk/README.md) now
include reconstruction latent reuse. These are provisional candidates, not final
dual-platform acceptance or implementation of every proposed optimization.

## Decision: one model, two runtime adapters

Keep model semantics in Rust, generic over a small backend interface with an
associated tensor type. Choose the backend once at the application boundary;
monomorphize hot model paths rather than dispatch every operator through trait
objects. The interface covers only operations proven necessary by NCM, not a
new general-purpose tensor framework.

```text
ncm-cli
  ├── ncm-io                 manifests, weights and artifact verification
  ├── ncm-tokenizers         text IDs, special tokens, byte BPE, Sudoku
  ├── ncm-model              codec, generator, sampler, request state
  │     └── ncm-core         validated config, backend contract, typed errors
  ├── ncm-backend-mlx ──────> ncm-core + mlx-rs -> mlx-c -> MLX/Metal
  └── ncm-backend-cuda ─────> ncm-core + Candle -> CUDA/cuBLAS/custom kernels
```

This is the intended full dependency graph; codec/backend/config edges are wired,
while the generator, sampler and inference CLI remain future work. `ncm-io` and tokenizers may depend on core types;
core never depends on a concrete runtime. A bounded Candle CPU implementation is
for fixtures and development, **not a third production runtime target**.

### Why these choices

- **MLX:** satisfies the explicit Apple runtime goal. Candle Metal or generic
  wgpu are not substitutes. Prefer existing safe `mlx-rs` wrappers; isolate any
  unavoidable custom FFI in the backend with documented safety invariants.
- **Candle CUDA:** Rust model/tensor logic without LibTorch; established loading
  and CUDA primitives. Use its existing CUDA interop rather than layering a
  second independently owned device/stream stack on top.
- **Not tch-rs:** would retain LibTorch, contrary to the port's motivation.
- **Not a new Burn/MLX backend:** excessive framework integration before proving
  the relatively small NCM operator set. Revisit only if capability spikes fail.
- **No Python production bridge:** Python is an offline oracle/export tool.
  A small, reviewed native kernel is preferable to recreating an optimized
  vendor matmul stack merely to claim 100% Rust.

## Core types and invariants

Plan typed `ModelConfig`, `CodecConfig`, `LevelSchedule`, `SamplingConfig`,
`DTypePolicy`, `ArtifactManifest`, and `CompatibilityProfile`. Use enums for
level-zero and output-head variants instead of contradictory boolean switches.
Configurations carry schema and upstream revisions, not an untyped Hydra map.

Validate before device allocation: increasing/nonempty levels, terminal latent
length, codebook/head counts, dimensions, even RoPE head width, attention head
ratios, normalization groups, vocab/PAD/EOS/code ranges, convolution divisibility,
prefix lengths, sampling ranges, checked byte-size arithmetic and memory budgets.
Upstream compatibility fields are translated explicitly; unknown inference-critical
fields error out. Unsupported research branches produce actionable errors.

Pure host configuration/I/O uses standard `Send`/`Sync` where appropriate; do not
impose these bounds on every tensor or add unsafe implementations to force MLX
arrays into a server design. Use device-owning workers if required by the runtime.
Keep caches, RNG state, accumulators, cancellation and scratch buffers in a
request/session object; model weights are immutable after load. Do not reproduce
upstream's model-global `_cfg` sampling state.

## Tensor contract (to validate before freezing)

- Logical activations BLC; attention layout explicit at the adapter boundary.
- Shape/dtype/device introspection, checked reshape/transpose/slice/broadcast,
  gather/scatter, embedding lookup, matmul/linear, reductions and activations.
- LayerNorm/RMSNorm/GroupNorm with explicit epsilon, axes and accumulation dtype.
- Conv1d/ConvTranspose1d, exact adaptive-area downsample and half-pixel linear
  upsample. Explicit weight-layout translation, tested with asymmetric kernels.
- Attention with noncausal target blocks, prefix padding, split-half RoPE, optional
  GQA; dense correct fallback before fused kernels.
- Sorting/top-k/cumulative sums and controlled sampling; no undocumented CPU
  fallback in a GPU adapter. Capability errors identify missing operations.
- Explicit execution/materialization/synchronization for MLX laziness and CUDA
  asynchronous launches, memory accounting and reproducible benchmarks.

Prefer composed backend operations initially; fused attention/norm/resampling
are capability-specific overrides retaining the same tested semantics. Avoid
host-device copies inside the level loop. Precompute static level schedules,
positions, resampling indices/weights, and head constants at model load.

## Correct optimization opportunities

**Prefix cache, not token-history cache.** In the released block-diagonal model,
prefix hidden states do not depend on target levels. Compute per-layer prefix K/V
once; each level computes its own Q/K/V and attends to prefix + that level.
Discard target K/V after the level: previous levels are represented by the latent
accumulator, not an ever-growing autoregressive KV cache. Preserve pad masking,
RoPE resets, query-level-zero and CFG conditional/null branches. For WMT14,
prefix prefill also produces the level-zero prediction.

This decomposition also permits noncausal target attention to `[prefix; target]`
without the asymmetric combined prefix/target mask; padding still needs a correct
masked or packed variable-length kernel. Prove equivalence against uncached eager
execution before turning it on. Never silently substitute a causal LLM kernel.

- Keep full-resolution latent accumulator and scratch allocations device-resident.
- Bound MLX lazy graphs via measured evaluation boundaries; compile per stable
  level/batch shape only after controlling recompilation and retained graphs.
- Benchmark CUDA graph capture only with stable addresses/shapes/RNG behavior;
  graph capture is not required for the initial implementation.
- Reuse codebook norms and weight transposes; keep GQA K/V compact if enabled.
- Eliminate full vocabulary sorting when semantics permit (e.g. top-p=1); keep
  a reference sampler and test tie behavior. Do not equate top-k=1 with greedy
  without accounting for tied maxima.
- Preserve final sequential token penalties. CPU f64 may be a compatibility mode;
  any FP32 device fast path is separately quality-gated and its transfer savings measured.
- Mixed precision, quantization and custom CUDA/Metal kernels are later, opt-in
  profiles. Retain FP32 reductions/latent accumulation where parity requires it.

## Platform/build policy

Phase 1 added optional, target-specific backend features `mlx` and `cuda`.
CLI runtime wiring is still future work. Default CPU-only checks require neither
a CUDA SDK nor MLX.
Build each platform separately rather than treating `--all-features` on Linux
as a valid cross-platform check. Unsupported target/runtime combinations fail
clearly. A CPU test path must not be mistaken for actual CUDA coverage.

Pin supported Rust crate versions and native runtime tuples after spikes; record
compiler, SDK, driver, GPU architecture and library versions with results. Start
CUDA validation on the available Ampere GPU; verify flash kernels' capabilities
rather than assume an H100-only optimization works there. Apple qualification currently covers macOS 26.7.3 / Xcode 27.2 with prebuilt MLX 0.32.2;
the selected wheel requires macOS 26.2+, but older tuples have not been tested.

Keep `unsafe_code = deny` by default. Any necessary audited FFI/mmap exception is
narrowly scoped with lifetime/ownership tests, never a blanket project-wide allow.
Use `Result`/typed errors, avoid panics on external data, and keep async orchestration
out of compute kernels. No `panic=abort` profile until deployment needs justify it.

## Compatibility is an architectural concern

Canonical artifact schemas and reference fixtures are versioned independently of
Rust packages and upstream Git. Upstream-specific import logic is an adapter, not
scattered conditionals across kernels. Preserve the last validated upstream profile
while qualifying the next one. See [maintenance policy](UPSTREAM.md); a dependency
upgrade is not evidence of upstream NCM compatibility.
