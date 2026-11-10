# Published comparison index

| Report | Scope | CUDA | MLX | Model quality |
| --- | --- | --- | --- | --- |
| [2026-11-09: operator capability spike](2026-11-09-cuda-operators/README.md) | 18 synthetic cases; three fresh-process timing runs per runtime | Measured, parity passed; slower composed resampling reported | Pending Linux gate; Mac now available | Not measured |
| [2026-11-10: Sudoku artifacts and oracle](2026-11-10-sudoku-artifacts/README.md) | 238 tensor roundtrips, tokenizer goldens, 56 Python model traces | Byte-exact conversion/readback; load performance/memory pending | Pending Linux gate | Conversion identity only; task quality and Rust model parity pending |

These are Phase 1 checks, **not** NCM generation speedups. Scoped Sudoku conversion
and reference traces exist; the general runtime importer, load-performance/memory
qualification, Rust model parity and Apple pass remain incomplete.

Follow [REPORTING.md](../REPORTING.md) for immutable bundles, affected-change
reports and release publication. Summaries, compact raw data and source hashes
are public; larger future bundles need durable release-asset links.
