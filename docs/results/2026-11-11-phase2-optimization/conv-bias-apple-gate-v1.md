# Decoder candidate: Apple gate with observed desktop activity

The first full Apple gate for candidate `b0a0058` remains **74/78** after its
predeclared, independent all-workload control; the initial matrix was 75/78.
The full `tools/check.sh --mlx` passes on real Metal, as do a fresh 642-trace /
2,500-decision trained qualification (zero ID disagreements, two near ties) and
all four real-asset tokenizer probes. Nine lifecycle processes agree on output
IDs. The first check attempt stopped because noninteractive SSH lacked the
Homebrew PATH; its log and the successful documented environment are retained.

The approved Qwen decoder improvement resolves the two previous Qwen failures:
controlled decode and skip-special median ratios versus accepted Rust are
0.69218 and 0.71914, with median interval upper bounds 0.71148 and 0.74802.
The following four controlled cells miss unchanged process-interval limits:

| Cell / comparator | Median ratio | p95 ratio | Failing upper bound |
| --- | ---: | ---: | ---: |
| Batch-4 resident block / Rust | 1.02092 | 1.05661 | p95 1.17571 > 1.10 |
| Batch-4 resident block / Python | 1.00684 | 1.01886 | p95 1.10748 > 1.10 |
| BERT decode / Rust | 1.00199 | 1.00360 | p95 1.10102 > 1.10 |
| TinyStories decode / Rust | 1.02505 | 1.02169 | median 1.05461 > 1.05; p95 1.19572 > 1.10 |

The initial matrix collected 108 activity samples; WindowServer appeared in
7 of the busiest-process samples, median 2.0% and peak 4.2%, and Chrome in 38,
peak 5.9%. In the 319 control samples, WindowServer appeared in 177, median
25.1% and peak 44.1%, and Chrome in 266, peak 95.5%. Samples capture only the
12 busiest processes; percentages are sampled CPU, not per-request GPU usage.
No thermal warning was reported before or after either run. Activity does not
prove the cause of the failures; it shows the control was not a quiet desktop.
No clock or power policy changed.

[Raw reports, plan, checks, telemetry, qualification and audits](raw/conv-bias-apple-gate-v1.tar.xz)
have SHA-256 `17a48aac5be1d64f3eaa326df8ce49d9487de1c22ae85971239ae066f01ce2c4`.
All 120 native inventories and 180 shared Python report copies match clean
baseline/candidate exports. macOS AppleDouble transfer sidecars were removed
from the local copies; report bytes were retained and verified. The controlled
audit SHA-256 `e6bad57541ec60f71716481921a0fb9ff12ea3bd8d122eac1a1e0900845d6dba`
reproduces the four failures byte-for-byte from a clean source export. The user
then confirmed Chrome closed and active desktop work paused; a separately
predeclared full quiet-window matrix and all-workload control are underway.
