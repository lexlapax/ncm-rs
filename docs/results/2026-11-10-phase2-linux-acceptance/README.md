# Extended Phase 2 Linux acceptance

**Linux/CUDA extended gate passes for the measured Phase 2 matrix.** Apple
remediation is next; neither Phase 2 as a whole nor Phase 3 is cleared.

The production implementation and full quality/latency/throughput/memory tables
remain in [CUDA remediation](../2026-11-10-phase2-cuda-remediation/README.md).
No production model, precision, oracle, tolerance, artifact or tokenizer changed
in this follow-up. The native/tokenizer implementation is `d561229`; new examples
and tools measure lifecycle and audit the existing matrix.

## Acceptance and retained uncertainty investigations

`acceptance.json` is generated from raw reports by
`tools/bench/audit_phase2_performance.py`. All **78 comparisons pass**: 29 latency
cells against both previous Rust and Python, plus peak observed process RSS/HWM
for codec and all four tokenizers against both. Limits remain median 1.05,
p95 1.10 and comparable host memory 1.05. CUDA allocator counters remain separately
reported, not mislabeled process/device peaks. The audit checks complete pinned
workloads, runtime/source identity, shared Python reports, unchanged 642-trace
qualification and 2,500 exact decisions, and complete lifecycle evidence.

For ordinary three-process cells, the observed process-median envelope must also
fit 1.05. Noisy execution cells use documented controlled follow-ups, preserving
original failing measurements rather than erasing them:

- TinyStories: fixed **30 fresh triplets**, CPU 2, unchanged public work. Upper
  paired-process 95% bounds (median/p95): encode 1.0053/1.0057, decode
  1.0256/1.0541, skip-special 1.0312/1.0425. This resolves the original nine-triplet
  interval crossing 1.05.
- GPT-2: **nine triplets**, same CPU affinity. All three execution scopes pass
  point and uncertainty bands; largest median/p95 upper bounds 1.0233/1.0289.
- Qwen: the initial nine-triplet control remains negative: skip-special upper
  bounds 1.0629/1.1616. Old/new native tokenizer binaries are byte-identical.
  A fixed **30-triplet executable-identity control** additionally removes the
  different executable path. The runner refuses this mode unless binaries hash
  identically and workloads are tokenizer-only; both labels still run in separate
  fresh processes with their respective source checkouts. All scopes pass:
  largest median/p95 upper bounds 1.0216/1.0261. This establishes non-inferiority
  for unchanged implementation work, not a claim that a tokenizer optimization
  occurred or that executable-path effects are proven causal.

Every triplet rotates previous/Python/candidate order. Each process has five
warmups and thirty trials. Bootstrap resamples **whole paired processes** (10,000,
seed 20261110), not individual trials; intervals are descriptive per-cell bounds,
not simultaneous statistical guarantees. Original uncontrolled results and all
controls, including negatives, remain available. No threshold was widened.

## Fresh-process lifecycle

A separate diagnostic measures exactly one checked load and solved-board B1
reconstruction, retaining snapshots and checking the same output hash across all
three implementations. It is not the steady-state benchmark or disk-cold latency.
Three fresh processes/runtime; no cache flush or system-policy change.

| Median ms | Previous Rust | Candidate Rust | Python |
| --- | ---: | ---: | ---: |
| Parent spawn → imports/linker/context ready | 163.89 | 159.92 | 999.66 |
| Internal device-context initialization | 121.61 | 119.68 | 123.47 |
| Checked load | 234.74 | 232.75 | 351.07 |
| First reconstruction, identity validation included | 83.45 | 66.76 | 292.98 |
| Output/model drop and synchronization | 1.09 | 0.09 | 0.07 |
| Final marker → process exit/context destruction | 100.45 | 113.47 | 434.04 |
| Full process | 588.16 | 575.31 | 2074.20 |

Medians of separate intervals do not sum exactly. Parent times include observation
and process overhead; child context excludes imports/linking. Final exit is noisy
and not inferred from model-drop timing. Startup/exit results are descriptive,
not additional steady-state non-inferiority claims. Both lifecycle captures are
retained; v2 uses a temporary stderr file to avoid pipe backpressure. The old and
new Rust binaries use the same diagnostic source, separately compiled against
`0da7fbe` and the current CUDA implementation. No inference CLI is introduced.

## Reproduction and validation

Extract the existing `threeway-v2.tar.gz` from the CUDA remediation report into
`reports/phase2-performance-linux-threeway-v2`. Extract this report's three control
archives into corresponding `reports/phase2-performance-linux-<archive-stem>`
directories, and copy `raw/phase2-linux-lifecycle-v2.json` into `reports/`.
Then run, with a new output path:

```sh
python3 tools/bench/audit_phase2_performance.py \
  docs/results/2026-11-10-phase2-linux-acceptance/raw/config.json reports/rechecked-acceptance.json
```

The audit's inputs and source digest are recorded in its output. Regeneration
needs only committed reports and the standard Python library, not weights/GPU or
ignored reference captures. Lifecycle binaries/scripts and the trusted-command
runner are in source; commands, binary hashes and phase records are in raw JSON.

`NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`
passes, including 22 report-tool tests, real-CUDA primitives/components/tiny codec,
resampling/pool tests, formatting, Clippy and docs (`raw/cuda-check.log`). The
unchanged full trained qualification remains passing. New tests reject missing
lifecycle phases, bad identities, empty/failed traces, changed oracle/tolerances
and missing selections. No Python enters production inference.

Next: recheck Mac availability/state, preserve its separate candidate worktree,
pull reviewed pushed commits, diagnose host allocation amplification and checked
load/tokenizer deficits. Validate any shared fix on Linux before Apple testing.
