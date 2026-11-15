# MLX unmasked attention: Metal result

The isolated candidate passed the focused real-Metal test and `tools/check.sh
--mlx`, including 18 Metal operator cases. Trained capture produced 642 traces;
qualification against the existing CUDA oracle passed all traces with 2500
selection decisions and zero disagreements (two near ties). The six benchmark
reports have matching ID-output hashes in each paired scope.

The diagnostic alternated three fresh-process pairs. Each table entry is the
candidate/current ratio for the process median / nearest-rank p95 of 30 trials;
lower is faster. This is a diagnostic, not the Phase 2 acceptance gate.

| Batch | Scope | Pair 1 | Pair 2 | Pair 3 |
|---|---|---:|---:|---:|
| 1 | encode | 1.272 / 1.164 | 0.998 / 1.104 | 1.048 / 1.008 |
| 1 | decode | 1.148 / 1.203 | 1.128 / 1.140 | 1.031 / 1.065 |
| 1 | reconstruct | 0.997 / 0.989 | 1.015 / 1.036 | 0.986 / 0.992 |
| 1 | resident_decoder_block | 0.878 / 0.959 | 1.041 / 1.039 | 0.929 / 0.955 |
| 2 | encode | 0.966 / 0.848 | 0.796 / 1.048 | 1.143 / 0.947 |
| 2 | decode | 0.965 / 0.839 | 1.071 / 0.929 | 0.867 / 0.762 |
| 2 | reconstruct | 0.959 / 0.787 | 0.763 / 0.789 | 0.779 / 0.947 |
| 2 | resident_decoder_block | 1.067 / 1.058 | 0.894 / 0.855 | 0.895 / 0.876 |
| 4 | encode | 1.577 / 0.658 | 1.150 / 1.305 | 1.106 / 0.845 |
| 4 | decode | 0.996 / 0.971 | 1.231 / 1.202 | 0.962 / 1.023 |
| 4 | reconstruct | 1.005 / 1.042 | 0.996 / 0.985 | 1.006 / 0.984 |
| 4 | resident_decoder_block | 0.965 / 0.980 | 0.975 / 0.974 | 0.987 / 1.056 |

First-execution ratios were 1.007, 1.006, and 0.998; final host high-water
ratios were 0.999, 1.000, and 1.000. The batch-4 block median improved in all
three pairs but the p95 increased in one, and several other scopes regressed
or varied markedly. CPU activity samples showed corespotlightd up to 147.2%,
mediaanalysisd up to 58.5% and WindowServer up to 39.7%; these are sampled
process CPU percentages, not evidence of GPU overlap or causation. Thermal
status reported no warnings. This noisy, mixed result does not establish a
useful full-codec improvement or explain the prior tail. Do not integrate the
candidate or claim the existing Apple gate passed.

The [archive](raw/mlx-attention-nomask-metal-v1.tar.xz) includes all six
reports, activity samples, commands, validation logs, trained qualification,
and trace metadata; the large trace tensor is retained locally and on the Mac.
Archive SHA-256: `9740f9212c69a2ccc9bae1771454a9d22370948196b2bd5b0a8970abbd619d9a`. The [experiment plan](mlx-attention-nomask-plan-v1.md)
retains the Linux evidence and reviewed patch. SSH uses the configured
`macstudio` host entry without an explicit identity agent.
