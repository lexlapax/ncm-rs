# Reference fixtures

`operators-v1.json` contains 18 small synthetic FP32 PyTorch 2.15.0 CPU cases for
matmul, conv/transposed conv, GroupNorm/LayerNorm/RMSNorm, area/linear resampling,
padded prefix attention, split-half RoPE and FP16 rounding. Inputs are fixed
values, not RNG-dependent. Metadata pins the oracle and upstream semantic reference.

Regenerate to a NEW file with `tools/reference/operators.py generate --fixture ...`
and review differences. Shapes, lengths, finite values, schema and upstream pin are
validated before loading the Rust fixture tensors. Test tolerances are fixed in
`crates/ncm-backend-cuda/tests/support/mod.rs`, not controlled by the fixture.

This is not a checkpoint/model fixture. Large traces, weights and licensed data
remain outside Git. See [testing policy](../../docs/TESTING.md) and the
[measurement report](../../docs/results/2026-11-09-cuda-operators/README.md).
