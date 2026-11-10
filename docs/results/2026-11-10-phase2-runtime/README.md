# Phase 2 checked shared runtime construction

Scope: configuration, inventory preflight and native weight construction/readback.
**Not forward parity, a performance comparison, or a platform acceptance gate.**
The shared codec's encode/decode implementation and affected-workload load/latency/
throughput/memory comparisons remain pending later Phase 2 milestones.

## Implementation

- `ncm-core`: strict typed upstream codec kwargs, immutable validated config,
  exact generated tensor inventories, input token/code validation and provisional
  FP32 backend contracts. No Send/Sync requirements or Python dependency.
- Initial family: uncompressed lookup encoder, group pre-quant normalization,
  one bidirectional Q/K-normalized decoder block, manual multiscale quantizer,
  learned level-zero cascades and unshared constant-ratio Phi refinements.
- Explicit limits: length/width at most 256, vocabulary/codebooks at most 65,536,
  2–32 levels, batch 1–128, config 1 MiB and runtime weight bytes 256 MiB.
  These are implementation bounds, not text-profile support or peak-RSS promises.
- `ncm-model::Codec`: host preflight checks the complete runtime inventory, F32
  shapes and finite values before any backend upload. Training buffers remain
  separate. Temporary endian-decoded host storage is one tensor at a time.
  Allocation failures propagate without changing devices. Artifact integrity is
  not authentication; callers must still trust the manifest as documented.
- Only codec fields are executed. Generator, evaluation and data-path metadata
  are explicitly ignored; the loader never opens those paths. Unknown codec or
  tokenizer fields are errors. Canonical artifact construction requires the
  Sudoku task/pre-tokenizer; explicit synthetic safetensors construction is separate.

## Checks actually run

```sh
cargo test -p ncm-core -p ncm-model
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo run --release -p ncm-backend-cuda --features cuda \
  --example checked_codec_load -- artifacts/canonical/sudoku-easy-qualified cpu
./target/release/examples/checked_codec_load artifacts/canonical/sudoku-easy-qualified cuda
```

Full local checks pass: 22 Rust tests, seven report-tool tests, all 18 unchanged
real-CUDA operator cases, formatting, Clippy, docs and Python syntax. Seven new
Rust tests cover both reference inventories, malformed/unsupported configuration,
duplicate fields, per-level ID bounds, malformed payloads, wrong/missing/extra
weights, nonfinite data, zero allocations before failed preflight and backend
error propagation. Mock construction is not described as numerical execution.

On the Phase 1 Linux machine (RTX 3060 Ti; Rust 1.99.1, Candle 0.11.1), both CPU
and CUDA construct the checked shared codec with all **58 runtime tensors /
59,586,048 bytes** read back byte-exact against the pinned manifest. The generated
inventory also matches every tiny synthetic fixture tensor, including normalized
cascade stage indexing. Raw summaries are in `raw/load-cpu.json` and
`raw/load-cuda.json`; source is the accompanying commit. Artifact and upstream
pins are unchanged from [the reference contract](../../PHASE2.md).

No MLX Phase 2 work was started. No inference output or performance improvement
is claimed. Linux model parity and measurements must precede the Apple pass.
