# Phase 1 acceptance — complete

Accepted 2026-11-10: **Linux/CUDA first, then Apple Silicon/MLX**, on real hardware.
Phase 2 has not started. This closes the reference/artifact/runtime-feasibility
phase, not model inference, release qualification, or paper-quality reproduction.

## Gate evidence

| Requirement | Linux/CUDA | Apple/MLX |
| --- | --- | --- |
| Pinned reference / portable fixtures | NCM `c702742…`, HF `d342ce8…`; fixed operator and Sudoku tokenizer fixtures | Same fixtures, pins and tolerances; no regenerated expected outputs |
| Canonical weight roundtrip | All 238 Sudoku-easy tensors exact; 36 training buffers retained separately | Same manifest; all 238 copied MLX arrays read back exactly |
| Tokenizer IDs / decoding | Five upstream golden cases, invalid-input regression tests | Same tests pass; independent architecture timing/parity runs |
| Dense uncached model reference | 56 original-vs-converted CUDA traces exact; derived buffers inventoried | Portable trace archive available; no claim of a Rust/MLX model implementation |
| Runtime operators | 18 CPU and CUDA cases pass | All 18 real-Metal cases pass, including prime-stride transposed convolution |
| Synchronization / baselines | Synchronized eager operator timings; three-process load/tokenizer baselines | Forced evaluation; three-process adapted Python MLX vs Rust MLX operator/load/tokenizer baselines |
| Loading / memory behavior | Copied validated loader; host RSS and global driver snapshots | Checked unified-memory loader; MLX active/peak/cache and process RSS measured |
| Local checks | `tools/check.sh --cuda` passes | `tools/check.sh --mlx` passes |
| Public evidence | Immutable reports and raw measurements | Reports, raw trials, native pins/hashes, build workaround and limitations |

Evidence:
- [CUDA operator capability/baseline](results/2026-11-09-cuda-operators/README.md).
- [Artifacts and model traces](results/2026-11-10-sudoku-artifacts/README.md).
- [Linux loading/memory/tokenizer/compile check](results/2026-11-10-linux-baseline/README.md).
- [Apple capability and baseline](results/2026-11-10-mlx-phase1/README.md).
- [Durable 56-trace research asset](https://github.com/lexlapax/ncm-rs/releases/tag/phase1-fixtures-v1),
  with SHA-256 in the original oracle report. No weights/dataset bundled.

## Scope and capability decisions

- Phase 1's artifact prototype is explicitly **Sudoku-easy inspection v1**, not a
  universal runtime importer. Its checkpoints have no aliases or inference
  unigrams; unexpected metadata/sharing is rejected, not silently discarded.
  General resolved model configs, TinyStories unigram/byte-BPE assets and other
  profile imports are not advertised as implemented. See [artifact contract](ARTIFACTS.md).
- Use shared Rust semantics, Candle 0.11.1 CUDA and the pinned MLX adapter.
  Python remains offline only; both native runtimes execute without Python.
- Common logical layouts: BLC activations, BHLD attention; upstream conv weights
  converted explicitly at adapter boundaries. GroupNorm reduces sequence and
  grouped channels; exact area bins, half-pixel linear interpolation, split-half
  reset RoPE, and noncausal padded-prefix masks are covered by fixed fixtures.
- Keep FP32/eager compositions as the initial supported path. FP16 *operator*
  roundtrip passes; mixed-precision *models* are unqualified. Torch fullgraph compile
  fails the FP16-roundtrip golden and is not a supported compatibility profile.
- MLX evaluation boundaries are explicit and measured. No implicit CPU fallback,
  generic Metal replacement for MLX, or unsafe `Send`/`Sync` workaround was added.
  Shared fixture validation is now backend-independent. The production model
  backend trait remains intentionally provisional until Phase 2's real model code
  establishes the minimal API; no speculative framework was built.
- Native MLX 0.32.2 uses a separate project-local native prefix. The tested Mac lacks
  Xcode's optional Metal compiler; prebuilt support is why bindings use an immutable
  Git pin rather than the published crate. Older OS/native tuples remain untested.

## Negative findings and next-phase constraints

1. Candle's composed resampling is much slower than PyTorch's kernels. Preserve
   numerical correctness first; optimize only against measured real model shapes.
2. Apple Rust copied-load median is ~939 ms vs ~282 ms for adapted Python, and
   repeated-load host peak RSS is ~4.40 GiB vs ~0.41 GiB. MLX-managed live arrays
   return to zero after trials, but that does not establish a bounded host-memory
   envelope. Investigate before claiming an efficient production loader/server.
3. Benchmarks are workstation observations with warm filesystem caches and tiny
   operator shapes, not isolated cold-disk, model latency, or release memory budgets.
4. Full Rust codec/generator, official Sudoku quality, text profiles, request-safe
   caches, compiled/quantized inference and training are **not implemented or
   qualified**. OWT weights and the upstream full trainer remain unavailable.
5. Do not relabel the reference asset as an inference release or interpret Phase 1
   completion as model parity. Subsequent phases still require Linux then Mac and
   fresh affected-workload quality/performance evidence.

Next approved plan stage: Phase 2 shared primitives and codec, starting on Linux.
