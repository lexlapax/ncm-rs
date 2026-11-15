# Controlled Phase 2 performance summary

These tables describe the retained controlled audits, not a new run. The [current source and evidence binding](phase2-current-bindings-v2.md) checks all 78 cells per platform. The current Linux native CUDA pool is below 1.05× prior Rust. The user accepted the specific Apple immediate B4 live-memory exception; the [final audit](phase2-final-audit-v1.md) separately discloses the higher encode active-memory peak.

Timings are pooled controlled process trials in milliseconds; each cell has 30 trials per process (nine processes per runtime on Linux, 18 on Apple). Rate is calculated from the candidate median: codec batch items/s or tokenizer fixture strings/s. Checked-load rows have no rate. This is not generated-token throughput. Confidence bounds and every original result remain in the linked audit JSON.

## Linux/CUDA

[Complete controlled audit](cuda-bounded-linux-acceptance-v1.json).

| Workload | Candidate median / p95 ms | Candidate rate | Previous median / p95 ms | Python median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| codec/1/encode | 1.8836 / 1.9039 | 530.9 items/s | 2.0651 / 2.0878 | 3.5850 / 3.6033 |
| codec/1/decode | 1.1024 / 1.1234 | 907.1 items/s | 1.2443 / 1.2640 | 1.5331 / 1.6783 |
| codec/1/reconstruct | 2.0622 / 2.0870 | 484.9 items/s | 3.2823 / 3.3163 | 4.6709 / 5.1180 |
| codec/1/resident_decoder_block | 0.1332 / 0.1483 | 7505.6 items/s | 0.1771 / 0.1986 | 0.2599 / 0.2977 |
| codec/2/encode | 2.2142 / 2.2448 | 903.3 items/s | 2.4026 / 2.4287 | 3.4917 / 3.5925 |
| codec/2/decode | 1.4509 / 1.4702 | 1378.5 items/s | 1.6186 / 1.6300 | 1.7466 / 1.7652 |
| codec/2/reconstruct | 2.4217 / 2.4467 | 825.9 items/s | 4.0134 / 4.0509 | 5.1074 / 5.1584 |
| codec/2/resident_decoder_block | 0.1721 / 0.1740 | 11623.7 items/s | 0.2342 / 0.2365 | 0.2873 / 0.3071 |
| codec/4/encode | 2.8220 / 2.8408 | 1417.4 items/s | 3.0280 / 3.4825 | 3.9109 / 3.9420 |
| codec/4/decode | 1.9179 / 1.9316 | 2085.6 items/s | 2.1387 / 2.1808 | 2.1439 / 2.1597 |
| codec/4/reconstruct | 3.0873 / 3.1175 | 1295.6 items/s | 5.1490 / 5.2006 | 5.8857 / 5.9200 |
| codec/4/resident_decoder_block | 0.2286 / 0.2299 | 17495.5 items/s | 0.3165 / 0.3195 | 0.3173 / 0.3367 |
| codec/checked_load | 180.2724 / 183.6402 | — | 182.1016 / 185.6106 | 271.8276 / 276.1492 |
| bert/encode | 0.9726 / 0.9917 | 91503.8 strings/s | 1.0272 / 1.0435 | 2.0969 / 2.1410 |
| bert/decode | 0.4653 / 0.4792 | 191292.7 strings/s | 0.4606 / 0.4733 | 0.8112 / 0.8312 |
| bert/decode_skip_special | 0.3950 / 0.4039 | 225338.4 strings/s | 0.3894 / 0.3998 | 0.7338 / 0.7481 |
| bert/checked_load | 2.3798 / 2.4502 | — | 3.6828 / 3.8568 | 18.2911 / 19.0324 |
| gpt2/encode | 0.8892 / 0.9101 | 100091.7 strings/s | 0.9289 / 0.9522 | 2.1090 / 2.1618 |
| gpt2/decode | 0.0831 / 0.0882 | 1070438.5 strings/s | 0.1130 / 0.1174 | 0.4749 / 0.4882 |
| gpt2/decode_skip_special | 0.0922 / 0.0990 | 965596.6 strings/s | 0.1215 / 0.1263 | 0.4840 / 0.5017 |
| gpt2/checked_load | 16.3072 / 17.2598 | — | 17.2399 / 17.7654 | 52.0665 / 54.2287 |
| qwen3/encode | 0.9470 / 0.9851 | 93980.4 strings/s | 0.9816 / 1.0086 | 2.0923 / 2.1533 |
| qwen3/decode | 0.0735 / 0.0807 | 1211593.2 strings/s | 0.1022 / 0.1080 | 0.4553 / 0.4714 |
| qwen3/decode_skip_special | 0.0833 / 0.0908 | 1068889.3 strings/s | 0.1118 / 0.1185 | 0.4636 / 0.4755 |
| qwen3/checked_load | 59.5218 / 65.5268 | — | 61.6102 / 68.7976 | 151.8053 / 157.4468 |
| tinystories/encode | 0.2366 / 0.2423 | 376084.4 strings/s | 0.2360 / 0.2420 | 2.6176 / 2.7483 |
| tinystories/decode | 0.0132 / 0.0143 | 6740637.0 strings/s | 0.0150 / 0.0171 | 0.2832 / 0.2917 |
| tinystories/decode_skip_special | 0.0133 / 0.0144 | 6686199.4 strings/s | 0.0151 / 0.0168 | 0.5022 / 0.5164 |
| tinystories/checked_load | 2.8969 / 2.9629 | — | 3.3470 / 3.4283 | 28.0843 / 28.5097 |

First-load and first-execution values below summarize all controlled processes as median [minimum, maximum] in milliseconds. Process, OS disk and compiler caches were not cleared; process-local allocator handling follows the benchmark harness. First execution starts after loading and is not a disk-cold or compilation-only measure.

| Observation | Previous | Candidate | Python |
| --- | ---: | ---: | ---: |
| codec first load | 191.817 [189.993, 196.458] | 190.183 [188.854, 191.057] | 286.721 [284.590, 291.596] |
| bert first load | 5.010 [4.800, 5.357] | 4.099 [3.828, 4.519] | 24.899 [24.646, 25.449] |
| gpt2 first load | 19.422 [18.985, 20.023] | 19.481 [18.691, 19.699] | 62.890 [62.458, 65.131] |
| qwen3 first load | 72.401 [70.452, 77.853] | 71.239 [69.617, 75.482] | 178.360 [177.412, 183.833] |
| tinystories first load | 4.217 [4.168, 4.345] | 3.759 [3.676, 3.840] | 29.118 [28.861, 29.432] |
| codec first execution | 53.262 [52.368, 56.813] | 52.468 [51.901, 54.973] | 233.279 [228.095, 239.396] |

Separate lifecycle capture: three fresh processes per runtime; median [minimum, maximum] milliseconds. Parent startup includes process creation and initialization; parent exit is measured after the drop phase. These are not the controlled warm timings.

| Lifecycle observation | Previous | Candidate | Python |
| --- | ---: | ---: | ---: |
| parent startup | 151.755 [145.402, 236.889] | 144.860 [142.047, 146.190] | 895.187 [894.755, 908.955] |
| child load | 190.929 [190.776, 191.885] | 188.535 [188.139, 190.357] | 289.027 [285.199, 290.066] |
| child execution | 52.768 [52.089, 53.312] | 51.867 [51.637, 52.114] | 230.338 [229.215, 234.257] |
| child drop | 0.074 [0.072, 0.082] | 0.141 [0.137, 0.149] | 0.039 [0.038, 0.061] |
| parent post-drop exit | 109.516 [97.534, 114.221] | 98.431 [92.792, 141.391] | 345.297 [342.916, 347.068] |

Host memory is the maximum observed process HWM/RSS ratio from the three-process matrix, not the MLX allocator counter and not a separate physical-memory estimate.

| Workload | HWM / previous | HWM / Python | RSS / previous | RSS / Python |
| --- | ---: | ---: | ---: | ---: |
| codec | 1.0033 | 0.3697 | 1.0042 | 0.3585 |
| bert | 0.9015 | 0.1603 | 0.9015 | 0.1611 |
| gpt2 | 1.0016 | 0.2408 | 1.0016 | 0.2433 |
| qwen3 | 1.0017 | 0.3186 | 1.0017 | 0.3186 |
| tinystories | 0.9424 | 0.1411 | 0.9424 | 0.1411 |

## Apple/MLX

[Complete controlled audit](mlx-acceptance.json).

| Workload | Candidate median / p95 ms | Candidate rate | Previous median / p95 ms | Python median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| codec/1/encode | 4.2214 / 5.4341 | 236.9 items/s | 13.2170 / 13.9312 | 15.2460 / 16.1120 |
| codec/1/decode | 3.5507 / 3.9711 | 281.6 items/s | 12.0314 / 13.2164 | 13.8125 / 14.9628 |
| codec/1/reconstruct | 6.5479 / 7.4672 | 152.7 items/s | 24.9995 / 26.0742 | 28.7065 / 29.9101 |
| codec/1/resident_decoder_block | 0.8135 / 0.8714 | 1229.2 items/s | 1.1032 / 1.1639 | 1.1151 / 1.1910 |
| codec/2/encode | 4.8862 / 8.1133 | 409.3 items/s | 14.0343 / 14.7764 | 16.0847 / 17.1987 |
| codec/2/decode | 5.1101 / 7.4625 | 391.4 items/s | 13.0770 / 13.9238 | 14.9044 / 15.6774 |
| codec/2/reconstruct | 9.0945 / 10.1345 | 219.9 items/s | 27.0714 / 28.3553 | 30.8560 / 32.0805 |
| codec/2/resident_decoder_block | 0.8848 / 1.0691 | 2260.4 items/s | 1.1874 / 1.2418 | 1.1957 / 1.2528 |
| codec/4/encode | 7.7564 / 10.1881 | 515.7 items/s | 15.0647 / 16.1214 | 17.1227 / 17.9440 |
| codec/4/decode | 8.2335 / 9.4867 | 485.8 items/s | 14.2749 / 15.2549 | 16.0288 / 17.1125 |
| codec/4/reconstruct | 11.0123 / 12.0410 | 363.2 items/s | 29.2021 / 30.5480 | 33.0051 / 33.8256 |
| codec/4/resident_decoder_block | 1.1298 / 1.2780 | 3540.5 items/s | 1.2161 / 1.3431 | 1.2299 / 1.3139 |
| codec/checked_load | 174.8991 / 175.5832 | — | 179.9914 / 180.7427 | 270.6796 / 279.8151 |
| bert/encode | 1.2075 / 1.2514 | 73708.5 strings/s | 1.2767 / 1.3199 | 2.4110 / 2.4598 |
| bert/decode | 0.7120 / 0.7330 | 124992.6 strings/s | 0.7122 / 0.7373 | 1.0168 / 1.0331 |
| bert/decode_skip_special | 0.5940 / 0.6127 | 149821.3 strings/s | 0.5950 / 0.6132 | 0.8948 / 0.9081 |
| bert/checked_load | 2.7869 / 2.9030 | — | 3.4849 / 3.6120 | 17.5204 / 17.6857 |
| gpt2/encode | 1.1741 / 1.2908 | 75801.1 strings/s | 1.2345 / 1.3762 | 2.4655 / 2.5748 |
| gpt2/decode | 0.1182 / 0.1212 | 753174.5 strings/s | 0.1746 / 0.1779 | 0.5020 / 0.5129 |
| gpt2/decode_skip_special | 0.1257 / 0.1276 | 707984.3 strings/s | 0.1814 / 0.1849 | 0.5140 / 0.5235 |
| gpt2/checked_load | 15.8378 / 15.9928 | — | 17.0049 / 17.1908 | 51.1570 / 51.9439 |
| qwen3/encode | 1.2479 / 1.3471 | 71318.8 strings/s | 1.3308 / 1.4505 | 2.3823 / 2.4876 |
| qwen3/decode | 0.1126 / 0.1273 | 790233.1 strings/s | 0.1678 / 0.1738 | 0.4841 / 0.4957 |
| qwen3/decode_skip_special | 0.1309 / 0.1385 | 679822.6 strings/s | 0.1802 / 0.1897 | 0.5045 / 0.5160 |
| qwen3/checked_load | 59.5345 / 60.7960 | — | 63.8434 / 65.3840 | 146.1534 / 148.6266 |
| tinystories/encode | 0.3692 / 0.3965 | 241042.8 strings/s | 0.3666 / 0.3936 | 2.6688 / 2.6939 |
| tinystories/decode | 0.0170 / 0.0177 | 5248260.4 strings/s | 0.0203 / 0.0210 | 0.3538 / 0.3625 |
| tinystories/decode_skip_special | 0.0173 / 0.0182 | 5146888.7 strings/s | 0.0202 / 0.0215 | 0.6090 / 0.6245 |
| tinystories/checked_load | 3.4164 / 3.5466 | — | 3.7974 / 3.8737 | 30.8374 / 31.3325 |

First-load and first-execution values below summarize all controlled processes as median [minimum, maximum] in milliseconds. Process, OS disk and compiler caches were not cleared; process-local allocator handling follows the benchmark harness. First execution starts after loading and is not a disk-cold or compilation-only measure.

| Observation | Previous | Candidate | Python |
| --- | ---: | ---: | ---: |
| codec first load | 181.923 [180.224, 183.146] | 177.175 [176.085, 178.452] | 273.172 [237.451, 283.552] |
| bert first load | 4.208 [4.129, 4.466] | 3.751 [3.690, 4.156] | 27.135 [26.892, 28.236] |
| gpt2 first load | 18.156 [17.987, 20.101] | 17.202 [17.035, 17.321] | 62.352 [62.089, 65.584] |
| qwen3 first load | 67.348 [66.760, 68.005] | 63.099 [62.484, 69.441] | 160.384 [159.700, 162.075] |
| tinystories first load | 4.478 [4.312, 4.608] | 3.998 [3.947, 4.121] | 31.239 [30.928, 31.909] |
| codec first execution | 67.388 [58.304, 73.664] | 28.337 [26.770, 30.325] | 71.572 [70.087, 82.363] |

Separate lifecycle capture: three fresh processes per runtime; median [minimum, maximum] milliseconds. Parent startup includes process creation and initialization; parent exit is measured after the drop phase. These are not the controlled warm timings.

| Lifecycle observation | Previous | Candidate | Python |
| --- | ---: | ---: | ---: |
| parent startup | 39.676 [39.435, 41.713] | 40.022 [39.671, 531.065] | 106.127 [106.034, 106.145] |
| child load | 180.494 [180.259, 180.664] | 175.280 [175.130, 179.525] | 287.785 [279.842, 294.928] |
| child execution | 62.549 [61.811, 62.981] | 27.287 [27.124, 28.366] | 66.501 [59.313, 66.759] |
| child drop | 0.036 [0.033, 0.037] | 0.038 [0.025, 0.041] | 0.063 [0.063, 0.079] |
| parent post-drop exit | 7.662 [7.584, 8.245] | 5.805 [5.783, 5.823] | 18.457 [17.369, 18.646] |

Host memory is the maximum observed process HWM/RSS ratio from the three-process matrix, not the MLX allocator counter and not a separate physical-memory estimate.

| Workload | HWM / previous | HWM / Python | RSS / previous | RSS / Python |
| --- | ---: | ---: | ---: | ---: |
| codec | 0.9488 | 0.9022 | 0.9488 | 0.9033 |
| bert | 0.8627 | 0.1334 | 0.8627 | 0.1334 |
| gpt2 | 1.0026 | 0.1133 | 1.0026 | 0.1133 |
| qwen3 | 1.0005 | 0.2324 | 1.0005 | 0.2324 |
| tinystories | 0.9798 | 0.1214 | 0.9798 | 0.1214 |

Reproduce with `python3 docs/results/2026-11-11-phase2-optimization/phase2-performance-summary-v2.py`. The two controlled audit files and referenced archives are inputs; the script neither collects new measurements nor adjudicates live memory.
