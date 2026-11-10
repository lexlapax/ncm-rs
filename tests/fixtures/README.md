# Reference fixtures

`operators-v1.json` contains 18 small synthetic FP32 PyTorch 2.15.0 CPU cases for
matmul, conv/transposed conv, GroupNorm/LayerNorm/RMSNorm, area/linear resampling,
padded prefix attention, split-half RoPE and FP16 rounding. Inputs are fixed
values, not RNG-dependent. Metadata pins the oracle and upstream semantic reference.

Regenerate to a NEW file with `tools/reference/operators.py generate --fixture ...`
and review differences. Shapes, lengths, finite values, schema and upstream pin are
validated before loading the Rust fixture tensors. Test tolerances are fixed in
`crates/ncm-backend-cuda/tests/support/mod.rs`, not controlled by the fixture.

`sudoku-tokenizer-v1.json` contains five cases generated directly by the pinned
upstream Sudoku tokenizer, including BOS, zero/blank normalization, separators
and Python whitespace behavior. Rust encode/decode must match exactly; invalid
Rust token IDs are rejected rather than permitting Python's negative indexing.
Generate to a new file with `tools/reference/sudoku_trace.py`, never silently
replace the established fixture.

`codec-v1.json` is the Phase 2 **untrained synthetic codec** oracle: a width-16,
length-16 model with a normalized two-stage learned cascade, two input batches,
free-running encoding and fixed-code decoding. Its 170 deduplicated arrays use
F32/I64 little-endian hex bytes; weights/traces refer to their interned IDs.
The reference contains 236 traced tensors and per-position winner margins. It is
not trained Sudoku weights. See [Phase 2 policy](../../docs/PHASE2.md) and the
[oracle audit](../../docs/results/2026-11-10-phase2-oracle/README.md). Rust fixture
execution is pending the shared backend/model implementation.

The operator fixture is not a checkpoint/model fixture. The scoped Sudoku oracle
now generates 56 model traces from pinned weights; large traces, weights and
licensed data remain outside Git. See [testing policy](../../docs/TESTING.md) and the
[measurement report](../../docs/results/2026-11-09-cuda-operators/README.md).
