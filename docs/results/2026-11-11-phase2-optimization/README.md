# Additional Phase 2 optimization — baseline checkpoint

**In progress, not final acceptance.** The approved plan is in
[the five-step contract](../../PHASE2-OPTIMIZATION-REVIEW.md#approved-five-step-phase-2-plan-and-decision-gates).
This checkpoint freezes and refreshes the Linux baseline before production edits.
No new optimization is implemented here; no new Apple run has started. The earlier
accepted implementation remains unchanged. Phase 3 remains out of scope.

Subsequent provisional work is reported separately: [lower-risk candidates](../2026-11-11-phase2-low-risk/README.md)
and [prepared state/raw-upload PoC](../2026-11-11-phase2-prepared-state/README.md).
[Fixed Linux controls/resources](../2026-11-11-phase2-linux-controls/README.md)
now retain six open comparisons (72/78). These open gates and optional API review
do not amend this frozen baseline. The subsequent [single quiet-window diagnostic](linux-diagnostic-v2-results.md)
completed with all outputs/telemetry verified but unresolved B1 block recurrence;
it does not replace acceptance. See the [working completion audit](completion-audit.md).

## Baseline identity and locked gates

- Plan: `013f745`; clean baseline worktree: `3307dff`; production behavior: `083c1e5`.
- Rebuilt `codec_bench` and `asset_bench` **exactly match** the executable hashes in
  the accepted Linux capacity-tokenizer matrix. Saved immutable copies of those
  binaries plus codec profile/trace/lifecycle and tokenizer probe in
  `.reference/phase2-optimization-baseline-bin`; no historical worktree overwritten.
- [baseline.json](baseline.json) records source/executable/evidence hashes and
  limitations. [acceptance-v1.json](acceptance-v1.json) locks the approved unchanged
  matrix and 1.05 median/memory / 1.10 p95 ceilings before production optimization,
  against both accepted Rust and qualified Python. This is not a new tolerance.
- Three fresh processes/runtime, five warmups, thirty trials, rotating three-way
  order, CPU affinity 2 for the execution matrix. Native previous/candidate
  executables are byte-identical in this baseline-only run. Python samples are
  shared between summaries, not counted as independent duplicates.

## Findings, including uncertainty

All benchmark output checks pass. Fresh baseline native capture passes **642
trained traces / 2,500 decisions**, with zero disagreements and unchanged budgets.
The cleared-environment tokenizer probe passes all four profiles and rejection
cases. `tools/check.sh --cuda` passes, including **29 report-tool tests** and the
real-device operator probe. These validate the baseline, not future candidates.

The diagnostic whole-matrix audit is **73/78**, not passing acceptance. All pooled
point estimates and comparable host-memory ratios pass; five process envelopes
versus the identical native baseline exceed 1.05:

| Cell | Envelope ratio |
| --- | ---: |
| GPT-2 decode | 1.05417 |
| GPT-2 skip-special decode | 1.05121 |
| Qwen decode | 1.07748 |
| Qwen skip-special decode | 1.05896 |
| Qwen checked load | 1.07033 |

Additionally, B1 reconstruction medians are 3.2931 / 2.9974 ms in the two native
placements despite identical executable bytes. That apparent 9% improvement is
**not an optimization**. Its cause is not established. Keep matched native
controls and process uncertainty analysis for actual changes; do not transplant
historical controls or attribute executable-placement/process noise to code.
Desktop/compositor activity and GPU processes were sampled and retained. No
unrelated process, clock, power policy or global tooling was changed.

Fresh diagnostic CUPTI capture reproduces **1,312 kernels / 33,746 driver events**
for checked loading plus two B1 reconstructions. Summed kernel time is 4.464 ms,
not request wall time. There are 1,088 H2D copies totaling 59,703,176 bytes in that
whole-process scope; much of the volume is weight loading, not repeated request
metadata. The profile supports investigating dispatch/immutable metadata and
reconstruction reuse, not claiming a measured optimization yet.

Nine fresh lifecycle processes have complete ordered context/load/execute/drop
phases and identical solved-board output hashes. Lifecycle was not CPU-pinned;
it remains separate from the execution matrix, including startup and process exit.

## CPU/resource diagnostic and measurement correction

`tools/bench/run_process_resources.py` runs rotating independent processes and
uses `wait4` for each child's CPU time, faults, switches and lifetime RSS counter.
It does not inject profiling, and does not report per-request CPU, allocations or
acceptance wall times. Three processes per runtime/profile are retained.

Whole-process median CPU time (user + system, milliseconds):

| Scope | Accepted Rust | Qualified Python |
| --- | ---: | ---: |
| Codec benchmark process | 8839.08 | 14098.42 |
| BERT benchmark process | 255.72 | 1405.66 |
| GPT-2 benchmark process | 779.82 | 2722.49 |
| Qwen benchmark process | 2869.94 | 7024.98 |
| TinyStories benchmark process | 141.30 | 1729.15 |

These totals include startup, repeated loading, all scopes/trials, reporting and
teardown. They cannot identify an individual operator's CPU cost. Focused host
allocation diagnostics remain pending before causal allocation claims.

The first resource run exposed a counter-coverage trap: native small-tokenizer
`wait4` RSS showed an identical ~31.3 MB floor while the child's own HWM was lower
(e.g. TinyStories 10.96 MB). Pre-exec inherited high-water state can affect the
lifetime counter. Version 2 streams executable fingerprint reads rather than
allocating entire binaries in the launcher and labels the counter
`wait4_peak_rss_bytes_including_pre_exec`. **Neither version's wait4 RSS substitutes
for benchmark-reported acceptance RSS.** Both raw runs and the v1 runner source
are retained. Five tool regressions cover ordering, independent-child counters,
unsafe/instrumented inputs, failed children, missing reports and overwrite refusal.

## Commands and reproduction

Executed on Linux, before any production edit:

```sh
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo build -p ncm-backend-cuda --features cuda --release --locked \
  --example codec_bench --example codec_profile --example codec_trace --example codec_lifecycle
CARGO_BUILD_JOBS=4 cargo build -p ncm-tokenizers --release --locked \
  --example asset_bench --example asset_probe
git worktree add --detach .reference/phase2-optimization-baseline 3307dff
# Copy built executables into a new baseline-bin directory; compare accepted hashes.
python3 tools/bench/run_phase2_linux_comparison.py \
  .reference/phase2-optimization-baseline .reference/phase2-optimization-baseline-bin \
  reports/phase2-optimization-linux-baseline-v1 --cpu 2
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
.reference/phase2-optimization-baseline-bin/codec_trace \
  artifacts/canonical/sudoku-easy-qualified reports/phase2-oracle/v1/sudoku-cuda \
  reports/phase2-codec/cuda-optimization-baseline cuda
.reference/venv/bin/python tools/reference/qualify_codec.py \
  reports/phase2-oracle/v1/sudoku-cuda reports/phase2-codec/cuda-optimization-baseline \
  reports/phase2-optimization-preflight/qualification.json
env -i PATH="$PATH" HOME="$HOME" .reference/phase2-optimization-baseline-bin/asset_probe \
  artifacts/canonical/tokenizers-qualified tests/fixtures/tokenizers-v1.json
```

The initial qualification invocation with system `python3` failed because Torch
was absent; the pinned reference venv invocation above succeeded. No dependency
or global environment was changed to fix that invocation.

Lifecycle commands are embedded in `raw/lifecycle.json`; CPU commands and process
resources are in the resources archive. CUPTI used process-local
`CUDA_INJECTION64_PATH=$PWD/.reference/phase2-cupti/libcupti_trace_injection.so`
and `taskset -c 2 .reference/phase2-optimization-baseline-bin/codec_profile
artifacts/canonical/sudoku-easy-qualified`. Instrumented timing is not acceptance.

Reproduce the audit/summaries in a clean checkout without weights, reference
venvs or GPU access (audit **must exit 1**, preserving the five uncertain cells):

```sh
D=docs/results/2026-11-11-phase2-optimization
mkdir -p reports/phase2-optimization-preflight
tar -xzf "$D/raw/linux-baseline-v1.tar.gz" -C reports
cp "$D/raw/qualification.json" "$D/raw/lifecycle.json" reports/phase2-optimization-preflight/
python3 tools/bench/phase2_summary.py reports/phase2-optimization-linux-baseline-v1/previous /tmp/previous-summary.json
python3 tools/bench/phase2_summary.py reports/phase2-optimization-linux-baseline-v1/candidate /tmp/candidate-summary.json
python3 tools/bench/audit_phase2_performance.py "$D/raw/baseline-diagnostic-config.json" /tmp/baseline-audit.json
```

Compare JSON to the committed summaries/audit. Extract
`raw/linux-resources-v1-v2.tar.gz` under `reports/` to inspect CPU diagnostics;
`baseline.json` pins every compact evidence file. Executables, weights and large
trained trace tensors remain outside Git.

## Remaining work

Baseline identity, first controlled matrix, fresh Linux correctness, lifecycle
and initial CPU/CUDA diagnostics are recorded. The five noise cells remain open;
no optimization or new platform gate is accepted. Next: focused allocation
attribution, lower-risk independent experiments, matched candidate controls, then
the full Linux gate before any Apple implementation/testing. Follow the plan's
remaining measurement, conditional-escalation and strict completion audit steps.
