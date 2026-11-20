# Published comparison index

## Phase 5: release candidate in progress

These reports are development evidence, **not** a Phase 5 completion or release claim. The
[interim step 1 audit](2026-11-19-phase5-cli/step1-interim-audit-v1.md) records unresolved
artifact and CLI decisions. Linux qualification, the Mac replay, full quality/performance evidence,
profiling and the release gates remain pending. The [current context map](../../agent-context-map.md)
tracks the latest state.

| Report | Scope | Status and limits |
| --- | --- | --- |
| [CUDA CLI routing](2026-11-19-phase5-cli/cuda-routing-screen-v2.md), [current CLI and XSum flags](2026-11-19-phase5-cli/cuda-current-routing-v1.md), [TinyStories five seeds](2026-11-19-phase5-cli/cuda-tinystories-five-seed-routing-v1.md), [TinyStories quality retest](2026-11-19-phase5-cli/cuda-tinystories-quality-retest-v1.md) | Fixed-profile routing, experimental XSum flags and five-seed TinyStories CUDA quality retest | TinyStories quality passes its scoped retest; performance, other quality and Metal gates remain open. |
| [Decode (including retained WMT14, XSum and TinyStories outputs)](2026-11-19-phase5-cli/cuda-decode-screen-v1.md), [text input](2026-11-19-phase5-cli/cuda-text-input-v1.md), [overlength CUDA text](2026-11-19-phase5-cli/cuda-text-truncation-v1.md), [LM1B empty text](2026-11-19-phase5-cli/cuda-lm1b-empty-text-v1.md), [LM1B-small quality](2026-11-19-phase5-cli/cuda-lm1b-small-quality-retest-v1.md), [LM1B-medium quality](2026-11-19-phase5-cli/cuda-lm1b-medium-quality-retest-v1.md), [LM1B-large quality](2026-11-19-phase5-cli/cuda-lm1b-large-quality-retest-v1.md), [WMT14 base quality](2026-11-19-phase5-cli/cuda-wmt14-quality-retest-v1.md), [XSum top-1 quality](2026-11-19-phase5-cli/cuda-xsum-quality-retest-v1.md) | Retained-output decoding, corpus encoding, CUDA WMT14 and XSum full-corpus retests | WMT14 base, XSum top-1 and all LM1B sizes pass scoped nonblind CUDA retests; performance and Metal checks remain. |
| [Linux fresh-clone setup](2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md), [fresh-clone check fix](2026-11-19-phase5-cli/linux-fresh-reports-fix-v1.md), [resumed nine-profile replay](2026-11-19-phase5-cli/linux-guide-resumed-v2.md), [single-commit Linux replay](2026-11-19-phase5-cli/linux-guide-single-commit-v3.md), [export investigation](2026-11-19-phase5-cli/linux-export-repro-v1.md), [Sudoku re-export CUDA screen](2026-11-19-phase5-cli/cuda-sudoku-reexports-v1.md), [Sudoku CUDA corpus comparison](2026-11-19-phase5-cli/cuda-sudoku-reexport-corpus-v1.md), [Sudoku CLI corpus regression](2026-11-19-phase5-cli/cuda-sudoku-baseline-corpus-v1.md) | Fresh downloads/exports, tokenizer setup, synthetic examples and full-corpus Sudoku comparisons | Sudoku re-export qualification and complete Mac guide verification remain open. |
| [TinyStories current Python/Rust timing](2026-11-19-phase5-cli/cuda-tinystories-performance-retest-v1.md), [XSum current Python/Rust timing](2026-11-19-phase5-cli/cuda-xsum-performance-retest-v1.md) | Three alternating processes, 5 warmups and 30 measured trials at batches 1/4/32 (also 128 for TinyStories) | Shared desktop, sampled GPU memory, model probes rather than CLI; not full release gate. |
| [CUDA cold Sudoku](2026-11-19-phase5-cli/cuda-sudoku-cold-screen-v1.md), [TinyStories](2026-11-19-phase5-cli/cuda-tinystories-cold-screen-v1.md), [XSum/WMT14](2026-11-19-phase5-cli/cuda-xsum-wmt14-cold-screen-v1.md), [LM1B small](2026-11-19-phase5-cli/cuda-lm1b-cold-screen-v1.md), [LM1B medium/large](2026-11-19-phase5-cli/cuda-lm1b-medium-large-cold-screen-v1.md) | Rust baseline/candidate fresh-process startup, loading and one request per profile | Exploratory active-desktop screens; no native device peak, Python comparison, quality or release performance gate. |

The [compact cold-screen aggregates](2026-11-19-phase5-cli/cuda-cold-aggregates-v1.json) reproduce
selected statistics from the seven retained raw result files. They are not a complete result bundle:
raw captures remain local and ignored, not durably published. For each profile, the stored raw-result
and GPU-CSV hashes identify `reports/phase5-cli-cold-<profile>-v1/` (Sudoku-easy uses `-v4/`). The
aggregates were checked against those hashes and independently recomputed from 90 non-warmup rows per
condition: median, nearest-rank p95, median absolute deviation, and median/maximum process peak RSS.
All 210 warmup/measured output hashes per screen agreed. To recheck the retained local captures,
with trusted stable input directories:

```sh
setpriv --inh-caps=-all python3 tools/bench/audit_phase5_cold_aggregates.py \
  --aggregate docs/results/2026-11-19-phase5-cli/cuda-cold-aggregates-v1.json \
  --aggregate-sha256 adff3aea3317dd3bd7af0886bf6d0d2d1dbf854f96286de7ec642e71248fd762 \
  --reports reports
```

The audit checks capture hashes, the recorded binary identities, trial inventory, stdout hashes
and selected recomputed statistics; it does not reverify binaries or establish measurement
validity, GPU isolation, quality or release qualification. Its source SHA-256 is
`d1e5b1ff869bdd0f2312cd314df57a3efcfec5da60c9a414df7bfc6f4ff9c4aa`; four negative-control tests
and `tools/check.sh` passed. See each report for commands, hashes, measurement scope and limitations.

## Historical phase evidence

The [Phase 4 final audit](2026-11-19-phase4-audit/phase4-final-audit-v1.md) records the
per-profile text evidence on CUDA and Metal and the accepted exceptions.
The [Phase 3 Sudoku report](2026-11-14-phase3-sudoku/README.md) records the
completed Linux/CUDA and Apple/MLX full-board quality, latency, throughput and
memory comparisons and final publication check. Entries below
retain their historical status at capture time.

| Report | Scope | CUDA | MLX | Model quality |
| --- | --- | --- | --- | --- |
| [2026-11-09: operator capability spike](2026-11-09-cuda-operators/README.md) | 18 synthetic cases; three fresh-process timing runs per runtime | Measured, parity passed; slower composed resampling reported | Pending Linux gate; Mac now available | Not measured |
| [2026-11-10: Sudoku artifacts and oracle](2026-11-10-sudoku-artifacts/README.md) | 238 tensor roundtrips, tokenizer goldens, 56 Python model traces | Byte-exact conversion/readback; load performance/memory pending | Pending Linux gate | Conversion identity only; task quality and Rust model parity pending |
| [2026-11-10: Linux load/memory baseline](2026-11-10-linux-baseline/README.md) | Three-process loader/tokenizer measurements, compile capability | Linux Phase 1 passed; compiled FP16 rounding unsupported | Subsequent report below | Not measured |
| [2026-11-10: Apple MLX phase gate](2026-11-10-mlx-phase1/README.md) | 18 operator cases, 238 tensors, load/memory/tokenizer comparisons | Prior Linux gate passed | Passed; slow/high-RSS Rust loading disclosed | Not measured |
| [2026-11-10: Phase 2 codec oracle](2026-11-10-phase2-oracle/README.md) | 236 tiny / 642 trained codec traces; declared budgets | CUDA reference captured; Rust pending; CPU/CUDA diagnostic failure retained | Phase 2 not started | Reference only |
| [2026-11-10: Phase 2 Apple runtime gate](2026-11-10-phase2-mlx/README.md) | 642-trace qualified adapted baseline; 30 matched fresh-process codec/tokenizer reports | Linux gate already passed | Measured; parity/output checks pass, high native host RSS retained | Not measured |

**Historical Phase 2 optimization sequence:** the earlier remediation was
reopened for the additional five-step plan approved on 2026-11-11. See the
[research and approved plan](../PHASE2-OPTIMIZATION-REVIEW.md), with
[source retrieval metadata](2026-11-11-phase2-optimization-review/sources.json) and
[static estimates](2026-11-11-phase2-optimization-review/derived.json). These are
research/planning artifacts, not new GPU measurements or acceptance. The
[additional optimization baseline checkpoint](2026-11-11-phase2-optimization/README.md)
now publishes refreshed Linux measurements, fresh baseline parity, CPU/CUDA/lifecycle
diagnostics and five unresolved uncertainty cells (73/78 diagnostic audit).
[Lower-risk Linux candidates](2026-11-11-phase2-low-risk/README.md) now show modest
tokenizer allocation reductions and faster reconstruction with unchanged parity;
their 74/78 diagnostic audit retains four uncertainty cells.
[Prepared state and isolated raw upload](2026-11-11-phase2-prepared-state/README.md)
add Linux correctness/cache-drop evidence and lower measured work counts, but
retain an unresolved B1 block timing regression; the public upload API awaits
review and production loading is unchanged.
[Fixed Linux controls and resources](2026-11-11-phase2-linux-controls/README.md)
retain all 360 control processes and six open comparisons (72/78); whole-model
drop/isolation and native retained-output savings are verified. No final new Linux
or Apple gate had yet been accepted at that checkpoint, and Phase 3 had not
started. Historical results below remain valid:
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
