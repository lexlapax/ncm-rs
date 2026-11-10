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

The operator fixture is not a checkpoint/model fixture. The scoped Sudoku oracle
now generates 56 model traces from pinned weights; large traces, weights and
licensed data remain outside Git. See [testing policy](../../docs/TESTING.md) and the
[measurement report](../../docs/results/2026-11-09-cuda-operators/README.md).
