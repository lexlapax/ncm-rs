# Decoder candidate: quiet-window Apple gate

After the [prior 74/78 Apple control](conv-bias-apple-gate-v1.md) recorded
substantial Chrome activity, the user confirmed Chrome was closed and active
Mac desktop work paused. A predeclared full matrix and independent all-workload
control for the same candidate `b0a0058` followed. The initial matrix passes
**72/78** and the fixed control **75/78** under unchanged limits. All prior
failed results remain retained; do not retry the unchanged candidate based on
this outcome. The separate nine-process lifecycle capture agrees on output IDs.

The three controlled cells still outside their limits are:

| Cell / comparator | Median ratio | p95 ratio | Failing interval upper |
| --- | ---: | ---: | ---: |
| Batch-4 resident block / Rust | 1.02745 | 1.00994 | p95 1.15021 > 1.10 |
| Batch-4 resident block / Python | 1.01553 | 1.07119 | p95 1.18129 > 1.10 |
| TinyStories decode / Rust | 1.02589 | 1.01955 | median 1.05625 > 1.05 |

Chrome did not appear among the 12 busiest processes in any of 106 initial or
320 control samples. WindowServer appeared in all; sampled median / peak CPU
was 18.45% / 35.0% initial and 21.5% / 43.3% control. These are sampled CPU
values, not per-request GPU usage or proof of exclusivity. No thermal warning
was reported before or after either run; no clock or power policy changed.
The failed result does not establish a cause for the remaining intervals.

[Raw reports, plan, activity, lifecycle, qualification and audits](raw/conv-bias-apple-quiet-gate-v1.tar.xz)
have SHA-256 `a6bbe7539ed2eda37387fb9634128832060978666468e6ee4c4ae20a947292f2`.
The controlled audit SHA-256
`246f6b80e4141317e77c80ffea9a3bb71a585c231e6d1c237f77748eddbe07bd`
reproduces byte-for-byte from a clean `b0a0058` source export. All 120 native
inventories and 180 shared Python report copies match clean source exports;
macOS AppleDouble transfer sidecars were removed, preserving report bytes.
Linux remains 78/78, while Apple and Phase 2 remain open. Next, investigate
substantive batch-4 block and TinyStories decode improvements, validate Linux
first for any shared change, and retain all existing evidence.
