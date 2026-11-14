# TinyStories loader candidate Apple gate

Candidate `3555c67` (runtime `5200924`) passed the initial **78/78** Apple
matrix, but its predeclared unconditional all-workload control passed
**76/78**. The remaining cells are batch-4 resident decoder block p95 versus
previous Rust and Python: paired-process interval uppers are **1.13695** and
**1.11834**, respectively, above 1.10. The controlled block median ratios are
0.92901 and 0.91851. The initial pass and controlled failure are both retained;
Phase 2 is incomplete, and the unchanged candidate must not be retried as a
substitute for a substantive fix.

TinyStories host RSS and HWM now pass, each at peak ratio **0.97674** versus
previous Rust. Its controlled checked-load median ratio is **0.89802** with p95
ratio **0.90905**. Full `tools/check.sh --mlx`, four cleared-environment real
tokenizer probes, and trained qualification pass: 642 traces, 2,500 decisions,
zero disagreements and two near ties. Nine lifecycle runs agree on IDs. The
codec benchmark binary is byte-identical to the prior gate candidate, whose
model runtime was tested in the [finite MLX memory observation](mlx-memory-addmm-v1.md);
this loader-only change does not alter that model runtime.

The initial matrix had 107 activity samples and the control 318. Chrome did not
appear in the top-process entries. WindowServer median/peak CPU was 12.9%/22.8%
initial and 12.9%/27.2% control. The aerial wallpaper and video decoder appeared
in all initial samples and in 120 control samples, ending before the late
candidate block process with the largest p95. These CPU samples do not establish
GPU exclusivity or a cause for individual outliers. No thermal warning was
reported before or after either collection; no clock or power policy changed.

[Raw reports, plan, check, probes, qualification, activity, lifecycle and audits](raw/tiny-borrow-apple-gate-v1.tar.xz)
have SHA-256 `3448f9bd9510bce353970802b57fbd42023855e4779798d4f58d08de4a3031d9`.
The controlled audit SHA-256
`195a750ccf82dbd360f162267f42c52c82cd6aa072a889c43a57106e26c3f658`
reproduces byte-for-byte from a clean candidate export. All 120 native report
inventories and 180 shared Python report copies match clean source exports; the
core trait omitted by the codec inventory is separately bound to the build.
