# Post-gate Apple CPU/resource comparison

If and only if the independent 18-process Apple control passes all 78 cells,
run the [separate resource diagnostic](apple-resources-v1.py). This completes
the Phase 2 requirement to report CPU work without contaminating acceptance
timings. It compares accepted Rust, retained candidate Rust and qualified Python
for the codec and four tokenizer profiles, with three fresh processes per role
and profile (45 total) in rotating order. The existing workload, assets and
benchmark binaries remain unchanged.

Preflight requires a passing 78-cell controlled audit and rehashes its recorded
inputs, all 270 control reports, the audit source, native executables and Python
package environments. Each resource child is measured using the existing
`wait4` harness. Its complete report and source inventory are checked against
the fresh gate's matrix, and the usual Metal codec/tokenizer output validation
must pass before a summary is produced. Retain every child report, exit, CPU
measurement, activity sample and failure. No acceptance measurement is replaced
or rerun by this diagnostic. The user's current Mac-ready window covers this
separate follow-up; if the gate fails, do not run it.

Report whole-process user+system CPU time, including startup, loading, all
benchmark trials, serialization and teardown. This is not per-request CPU,
GPU execution time, an allocation count, or independent energy measurement.
The `wait4` peak RSS may include inherited pre-exec state and is not substituted
for acceptance RSS. Prior Linux allocation profiles remain platform-specific;
if Apple allocation counts are unavailable, disclose that instead of inferring
them from CPU time or MLX allocator counters.

Launch with an otherwise empty environment and the pinned tokenizer Python:

```sh
env -i PATH=/usr/bin:/bin TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
  .reference/tokenizers-venv/bin/python \
  docs/results/2026-11-11-phase2-optimization/apple-resources-v1.py
```
