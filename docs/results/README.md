# Published comparison index

| Report | Scope | CUDA | MLX | Model quality |
| --- | --- | --- | --- | --- |
| [2026-11-09: operator capability spike](2026-11-09-cuda-operators/README.md) | 18 synthetic cases; three fresh-process timing runs per runtime | Measured, parity passed; slower composed resampling reported | Pending Linux gate; Mac now available | Not measured |
| [2026-11-10: Sudoku artifacts and oracle](2026-11-10-sudoku-artifacts/README.md) | 238 tensor roundtrips, tokenizer goldens, 56 Python model traces | Byte-exact conversion/readback; load performance/memory pending | Pending Linux gate | Conversion identity only; task quality and Rust model parity pending |
| [2026-11-10: Linux load/memory baseline](2026-11-10-linux-baseline/README.md) | Three-process loader/tokenizer measurements, compile capability | Linux Phase 1 passed; compiled FP16 rounding unsupported | Subsequent report below | Not measured |
| [2026-11-10: Apple MLX phase gate](2026-11-10-mlx-phase1/README.md) | 18 operator cases, 238 tensors, load/memory/tokenizer comparisons | Prior Linux gate passed | Passed; slow/high-RSS Rust loading disclosed | Not measured |
| [2026-11-10: Phase 2 codec oracle](2026-11-10-phase2-oracle/README.md) | 236 tiny / 642 trained codec traces; declared budgets | CUDA reference captured; Rust pending; CPU/CUDA diagnostic failure retained | Phase 2 not started | Reference only |

**Phase 2 is in progress**: its reference contract and
[checked shared-runtime construction](2026-11-10-phase2-runtime/README.md) and
[Linux component parity](2026-11-10-phase2-primitives/README.md) exist.
[Complete shared codec parity](2026-11-10-phase2-codec/README.md) now passes tiny
CPU/CUDA and all 642 trained CUDA traces. Tokenizer assets, performance/memory and
Apple codec checks remain pending; neither full Phase 2 runtime gate is closed.

**Phase 1 is complete**; see the [acceptance checklist](../PHASE1.md). Earlier rows
retain their historical scope. These are capability/baseline checks, **not** NCM
generation speedups or Rust model parity. The general runtime importer and full
model-quality/performance qualification remain later-phase work.

Follow [REPORTING.md](../REPORTING.md) for immutable bundles, affected-change
reports and release publication. Summaries, compact raw data and source hashes
are public; larger future bundles need durable release-asset links.
