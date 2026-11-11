# Published comparison index

| Report | Scope | CUDA | MLX | Model quality |
| --- | --- | --- | --- | --- |
| [2026-11-09: operator capability spike](2026-11-09-cuda-operators/README.md) | 18 synthetic cases; three fresh-process timing runs per runtime | Measured, parity passed; slower composed resampling reported | Pending Linux gate; Mac now available | Not measured |
| [2026-11-10: Sudoku artifacts and oracle](2026-11-10-sudoku-artifacts/README.md) | 238 tensor roundtrips, tokenizer goldens, 56 Python model traces | Byte-exact conversion/readback; load performance/memory pending | Pending Linux gate | Conversion identity only; task quality and Rust model parity pending |
| [2026-11-10: Linux load/memory baseline](2026-11-10-linux-baseline/README.md) | Three-process loader/tokenizer measurements, compile capability | Linux Phase 1 passed; compiled FP16 rounding unsupported | Subsequent report below | Not measured |
| [2026-11-10: Apple MLX phase gate](2026-11-10-mlx-phase1/README.md) | 18 operator cases, 238 tensors, load/memory/tokenizer comparisons | Prior Linux gate passed | Passed; slow/high-RSS Rust loading disclosed | Not measured |
| [2026-11-10: Phase 2 codec oracle](2026-11-10-phase2-oracle/README.md) | 236 tiny / 642 trained codec traces; declared budgets | CUDA reference captured; Rust pending; CPU/CUDA diagnostic failure retained | Phase 2 not started | Reference only |
| [2026-11-10: Phase 2 Apple runtime gate](2026-11-10-phase2-mlx/README.md) | 642-trace qualified adapted baseline; 30 matched fresh-process codec/tokenizer reports | Linux gate already passed | Measured; parity/output checks pass, high native host RSS retained | Not measured |

**The earlier Phase 2 remediation is accepted; Phase 2 is now reopened for the
additional five-step optimization plan approved on 2026-11-11.** See the
[research and approved plan](../PHASE2-OPTIMIZATION-REVIEW.md), with
[source retrieval metadata](2026-11-11-phase2-optimization-review/sources.json) and
[static estimates](2026-11-11-phase2-optimization-review/derived.json). These are
research/planning artifacts, not new GPU measurements or acceptance. Phase 3 is
unstarted and blocked on the expanded gate. Historical results below remain valid:
[earlier dual-platform acceptance](2026-11-10-phase2-acceptance/README.md), the
[extension contract](../PHASE2-PERFORMANCE.md) and [initial research/CPU PoC](2026-11-10-phase2-performance-research/README.md).
[Production CUDA remediation](2026-11-10-phase2-cuda-remediation/README.md) now
passes unchanged correctness and removes the codec latency deficit in three-way
measurements. [Extended Linux acceptance](2026-11-10-phase2-linux-acceptance/README.md)
now passes. [Shared loading](2026-11-10-phase2-shared-loading/README.md) and
[compact tokenizer structures](2026-11-10-phase2-compact-tokenizers/README.md)
retain Linux acceptance. [Apple remediation measurements](2026-11-10-phase2-mlx-remediation/README.md)
remove measured loading/memory deficits and pass all trained traces. Fixed
30-triplet controls and lifecycle evidence now close the final Apple gate:
**78/78 comparisons pass on each platform**, with all negative rounds retained.
The following are historical baseline acceptances, not acceptance of the new performance gate. Its reference contract and
[checked shared-runtime construction](2026-11-10-phase2-runtime/README.md) and
[Linux component parity](2026-11-10-phase2-primitives/README.md) exist.
[Complete shared codec parity](2026-11-10-phase2-codec/README.md) now passes tiny
CPU/CUDA and all 642 trained CUDA traces. [Pinned tokenizer assets](2026-11-10-phase2-tokenizers/README.md)
pass exact Linux parity. The [Linux Phase 2 runtime gate](2026-11-10-phase2-linux/README.md)
and [Apple Phase 2 runtime gate](2026-11-10-phase2-mlx/README.md) established the
original matched three-process codec/tokenizer latency, throughput and memory
baseline. Their deficits were remediated before the earlier Phase 2 closure;
the additional approved optimization now requires a separate closure.
The Apple codec/tokenizer [parity report](2026-11-10-phase2-mlx-codec/README.md)
retains the initial long-dot failure and FP32 projection correction; the final
Apple report retains high native host RSS and the adapted-baseline limitation.

**Phase 1 is complete**; see the [acceptance checklist](../PHASE1.md). Earlier rows
retain their historical scope. These are capability/baseline checks, **not** NCM
generation speedups or Rust model parity. The general runtime importer and full
model-quality/performance qualification remain later-phase work.

Follow [REPORTING.md](../REPORTING.md) for immutable bundles, affected-change
reports and release publication. Summaries, compact raw data and source hashes
are public; larger future bundles need durable release-asset links.
