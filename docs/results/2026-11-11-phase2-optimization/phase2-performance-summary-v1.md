# Controlled Phase 2 performance summary

These tables describe the retained controlled audits, not a new run. The independent [coverage and source binding](phase2-closure-bindings-v1.md) checks all 78 cells per platform. The separate comparable live-memory criterion remains unresolved.

Timings are pooled controlled process trials in milliseconds; each cell has 30 trials per process (nine processes per runtime on Linux, 18 on Apple). Rate is calculated from the candidate median: codec batch items/s or tokenizer fixture strings/s. Checked-load rows have no rate. This is not generated-token throughput. Confidence bounds and every original result remain in the linked audit JSON.

## Linux/CUDA

[Complete controlled audit](linux-acceptance.json).

| Workload | Candidate median / p95 ms | Candidate rate | Previous median / p95 ms | Python median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| codec/1/encode | 1.8826 / 1.9258 | 531.2 items/s | 2.0618 / 2.3883 | 3.5785 / 3.8469 |
| codec/1/decode | 1.1014 / 1.1219 | 907.9 items/s | 1.2421 / 1.4035 | 1.6614 / 1.6794 |
| codec/1/reconstruct | 2.0605 / 2.0958 | 485.3 items/s | 3.2751 / 3.3157 | 4.6767 / 5.1045 |
| codec/1/resident_decoder_block | 0.1458 / 0.1484 | 6860.1 items/s | 0.1773 / 0.1998 | 0.2598 / 0.2870 |
| codec/2/encode | 2.2103 / 2.2385 | 904.8 items/s | 2.3906 / 2.4533 | 3.4818 / 3.5176 |
| codec/2/decode | 1.4465 / 1.4714 | 1382.7 items/s | 1.6098 / 1.6463 | 1.7433 / 1.7673 |
| codec/2/reconstruct | 2.4158 / 2.6843 | 827.9 items/s | 3.9886 / 4.2570 | 5.0988 / 5.4647 |
| codec/2/resident_decoder_block | 0.1723 / 0.1739 | 11605.5 items/s | 0.2337 / 0.2364 | 0.2874 / 0.3153 |
| codec/4/encode | 2.8070 / 3.0953 | 1425.0 items/s | 3.0113 / 3.0751 | 3.9051 / 4.1495 |
| codec/4/decode | 1.9137 / 1.9474 | 2090.1 items/s | 2.1272 / 2.1540 | 2.1425 / 2.1729 |
| codec/4/reconstruct | 3.0789 / 3.1260 | 1299.2 items/s | 5.1243 / 5.4012 | 5.8892 / 6.3437 |
| codec/4/resident_decoder_block | 0.2280 / 0.2296 | 17546.2 items/s | 0.3159 / 0.3183 | 0.3199 / 0.3381 |
| codec/checked_load | 181.8545 / 185.0988 | — | 182.9830 / 185.9881 | 274.2282 / 279.0579 |
| bert/encode | 0.9683 / 0.9944 | 91915.1 strings/s | 1.0273 / 1.0521 | 2.1013 / 2.1509 |
| bert/decode | 0.4640 / 0.4860 | 191825.4 strings/s | 0.4595 / 0.4828 | 0.8080 / 0.8224 |
| bert/decode_skip_special | 0.3951 / 0.4114 | 225274.0 strings/s | 0.3896 / 0.4099 | 0.7313 / 0.7479 |
| bert/checked_load | 2.3914 / 2.4831 | — | 3.6986 / 3.8537 | 18.1588 / 19.2261 |
| gpt2/encode | 0.8907 / 0.9323 | 99922.6 strings/s | 0.9280 / 0.9528 | 2.1006 / 2.1489 |
| gpt2/decode | 0.0833 / 0.0917 | 1068203.0 strings/s | 0.1122 / 0.1175 | 0.4758 / 0.4980 |
| gpt2/decode_skip_special | 0.0922 / 0.1011 | 965538.9 strings/s | 0.1214 / 0.1276 | 0.4831 / 0.5053 |
| gpt2/checked_load | 16.5984 / 17.2352 | — | 17.2711 / 18.0261 | 52.1592 / 54.8276 |
| qwen3/encode | 0.9418 / 0.9694 | 94498.9 strings/s | 0.9834 / 1.0165 | 2.0815 / 2.1461 |
| qwen3/decode | 0.0736 / 0.0788 | 1209321.3 strings/s | 0.1025 / 0.1095 | 0.4499 / 0.4661 |
| qwen3/decode_skip_special | 0.0812 / 0.0867 | 1096126.6 strings/s | 0.1147 / 0.1203 | 0.4599 / 0.4855 |
| qwen3/checked_load | 59.5412 / 65.9679 | — | 61.7360 / 67.9147 | 151.0278 / 154.8578 |
| tinystories/encode | 0.2386 / 0.2475 | 372933.4 strings/s | 0.2353 / 0.2421 | 2.6185 / 2.6883 |
| tinystories/decode | 0.0132 / 0.0145 | 6718502.3 strings/s | 0.0151 / 0.0172 | 0.2841 / 0.2956 |
| tinystories/decode_skip_special | 0.0134 / 0.0143 | 6659682.7 strings/s | 0.0151 / 0.0166 | 0.5012 / 0.5158 |
| tinystories/checked_load | 2.8972 / 2.9681 | — | 3.3469 / 3.4354 | 28.1234 / 28.7134 |

First-load and first-execution values below summarize all controlled processes as median [minimum, maximum] in milliseconds. Process, OS disk and compiler caches were not cleared; process-local allocator handling follows the benchmark harness. First execution starts after loading and is not a disk-cold or compilation-only measure.

| Observation | Previous | Candidate | Python |
| --- | ---: | ---: | ---: |
| codec first load | 192.694 [192.368, 194.435] | 192.177 [190.853, 193.059] | 289.221 [286.403, 292.737] |
| bert first load | 4.855 [4.501, 5.225] | 4.174 [3.726, 4.620] | 24.754 [24.183, 25.414] |
| gpt2 first load | 19.905 [19.020, 20.843] | 19.075 [18.495, 19.456] | 62.902 [61.560, 64.139] |
| qwen3 first load | 71.771 [70.440, 73.053] | 71.320 [69.579, 72.559] | 179.055 [177.200, 181.476] |
| tinystories first load | 4.275 [4.142, 4.412] | 3.774 [3.678, 3.883] | 29.070 [28.754, 30.149] |
| codec first execution | 53.859 [53.259, 54.978] | 52.793 [52.222, 54.272] | 228.490 [223.104, 237.373] |

Separate lifecycle capture: three fresh processes per runtime; median [minimum, maximum] milliseconds. Parent startup includes process creation and initialization; parent exit is measured after the drop phase. These are not the controlled warm timings.

| Lifecycle observation | Previous | Candidate | Python |
| --- | ---: | ---: | ---: |
| parent startup | 150.402 [145.371, 283.681] | 150.597 [143.305, 152.839] | 918.243 [902.931, 922.375] |
| child load | 193.629 [193.400, 193.642] | 191.946 [190.799, 192.323] | 289.081 [287.324, 291.750] |
| child execution | 53.017 [51.899, 53.922] | 53.359 [53.200, 53.498] | 227.078 [226.732, 228.535] |
| child drop | 0.075 [0.071, 0.096] | 0.146 [0.141, 0.147] | 0.040 [0.037, 0.061] |
| parent post-drop exit | 107.508 [107.175, 145.965] | 92.010 [91.724, 95.370] | 340.771 [339.848, 350.086] |

Host memory is the maximum observed process HWM/RSS ratio from the three-process matrix, not the MLX allocator counter and not a separate physical-memory estimate.

| Workload | HWM / previous | HWM / Python | RSS / previous | RSS / Python |
| --- | ---: | ---: | ---: | ---: |
| codec | 0.9982 | 0.3676 | 0.9987 | 0.3562 |
| bert | 0.8997 | 0.1580 | 0.8997 | 0.1580 |
| gpt2 | 0.9999 | 0.2396 | 0.9999 | 0.2417 |
| qwen3 | 1.0014 | 0.3136 | 1.0014 | 0.3189 |
| tinystories | 0.9398 | 0.1409 | 0.9398 | 0.1409 |

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

Reproduce with `python3 docs/results/2026-11-11-phase2-optimization/phase2-performance-summary-v1.py`. The two controlled audit files and referenced archives are inputs; the script neither collects new measurements nor adjudicates live memory.
