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
execution now passes on CPU/CUDA, including complete free-running and fixed-code
paths; trained CUDA qualification is reported separately. The same fixtures and
trained capture now also pass [MLX qualification](../../docs/results/2026-11-10-phase2-mlx-codec/README.md);
Apple performance/memory acceptance remains pending.

`tokenizers-v1.json` contains the four pinned text-tokenizer profiles: 356 encode
cases and 2,314 decode comparisons across both special-token modes. It includes
synthetic Unicode/whitespace/special-token cases, byte decoder cases and seeded
mixed strings, not the downloaded vocabularies. Run `ncm-tokenizers`' `asset_probe`
against separately converted canonical assets. See the
[tokenizer report](../../docs/results/2026-11-10-phase2-tokenizers/README.md) for
exact versions, source locks, conversion safety and reproduction.

`codec-bench-v1.json` fixes the matched FP32 codec scopes, batch patterns, independent
snapshot retention and separate instrumented stages. `tokenizer-bench-v1.json`
defines the 89-case/profile scalar timing workload and canonical special IDs for
separate total/non-special throughput denominators. Neither changes parity goldens.
See the [Linux measurement gate](../../docs/results/2026-11-10-phase2-linux/README.md).

The operator fixture is not a checkpoint/model fixture. The scoped Sudoku oracle
now generates 56 model traces from pinned weights; large traces, weights and
licensed data remain outside Git. See [testing policy](../../docs/TESTING.md) and the
[measurement report](../../docs/results/2026-11-09-cuda-operators/README.md).
