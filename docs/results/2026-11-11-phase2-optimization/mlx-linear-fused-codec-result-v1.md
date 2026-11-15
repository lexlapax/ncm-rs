# Fused interpolation in the full codec: mixed, deferred

The isolated [patch](mlx-linear-fused-codec-v1.patch) keeps the interpolation
coordinates and weighted-sum formula and constructs a local custom-kernel handle
per call. Linux `tools/check.sh --cuda` passed, including 484 report-tool tests
and 18 real-CUDA operator cases. The first Linux attempt stopped before checks
because `sccache` was unavailable; the successful attempt disabled that wrapper
for the command. On the Mac, the focused operator suite passed, including a
noncontiguous view with NaN, infinities and signed zero; the full
`tools/check.sh --mlx` passed, including 18 real-Metal cases. The unchanged
trained CUDA oracle then qualified **642 traces and 2,500 selection decisions**,
with zero ID disagreements and two near ties.

Three alternating fresh-process pairs compared the candidate with the current
runtime. All twelve B1/B2/B4 scopes have matching input, supplied-code and
output-ID hashes; the existing checker also verifies logits against the unchanged
component budget. Source inventories are stable and differ only in the intended
MLX operator source. These measurements are diagnostic, not an acceptance gate.

| Scope | Fused/current paired process-median ratios | Pooled median / p95 ratios |
| --- | --- | --- |
| B2 encode | 0.6390, 0.6621, 1.2715 | 0.7384 / 0.9716 |
| B2 reconstruction | 1.1350, 0.8817, 1.2842 | 1.1127 / 0.9928 |
| B2 resident decoder block | 1.1667, 1.0152, 1.2749 | 1.1845 / 1.1976 |
| B4 encode | 0.5721, 0.9404, 0.8769 | 0.8435 / 0.9233 |
| B4 reconstruction | 1.1168, 1.1430, 0.9621 | 1.0746 / 0.9603 |
| B4 resident decoder block | 1.0236, 0.9496, 1.0078 | 1.0025 / 1.0638 |

The [offline reviewer](mlx-linear-fused-codec-review-v1.py) reproduces all 12
scopes and checks report/source/qualification hashes. Timing variation includes
the resident decoder-block scope, which does not call interpolation; these
results do not establish the cause of the changes. They also do not establish
a consistently useful full-codec improvement.

B4 encode maximum peak active MLX allocation fell from **155,857,880** to
**146,945,160 bytes**, about 5.7%; the earlier accepted Rust baseline measured
129,751,100 bytes in the matched live-memory diagnostic. B4 decode maximum
peak fell from 167,863,852 to 156,558,892 bytes. These are MLX active allocator
peaks, not host RSS or a comparable steady-state live-memory decision. The
candidate did not remove the outstanding immediate B4 decoder-block allocation
difference. Final process host HWM was 278,069,248–278,265,856 bytes current
and 278,249,472–278,282,240 bytes candidate. First executions were
26.82–30.31 ms current and 27.46–29.01 ms candidate; prior process and system
kernel caches were not cleared, so this is not a cold-start bound.

The 37 activity samples showed WindowServer median/max 0%/3.1% in the sampled
top entries and no Chrome entries; these do not establish GPU exclusivity.
No thermal warning was reported before or after. Preserve all measurements,
including unfavorable pairs. The candidate remains **deferred**: the peak
reduction is partial, full-codec timings are mixed, and safe general-purpose
integration, cache/initialization costs and the separate 1.05 live-memory
criterion remain unresolved. No production change or new gate is claimed.

[Mac checks, qualification and raw pairs](raw/mlx-linear-fused-codec-mac-v1.tar.xz)
have SHA-256 `a9e35e944b76f9fd1a7aef9fa06475b767f180396fdfe51a7a51facc03e2a918`.
The trace's large tensor file remains outside Git; its report and qualification
hashes are archived. The Linux logs are linked from the [plan](mlx-linear-fused-codec-plan-v1.md).
