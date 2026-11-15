# CUDA zero-workspace projection diagnostic

Three fresh processes compared strict-FP32 cuBLASLt fused projection and bias with zero versus 4 MiB workspace over 12 decoder shapes (5 warmups and 30 interleaved events each). Both modes were available in all cases, and all outputs passed the diagnostic's unchanged per-element tolerance against the 4 MiB mode; nine values per case were also checked against a CPU reference. The largest observed cross-mode absolute difference was 3.33786010742e-06. The greatest workspace requested by a selected 4 MiB algorithm was 2,187,264 bytes.

Median of the three per-process ratios of GPU-event medians (zero / 4 MiB):

| Batch | Projection | Ratio |
|---:|---|---:|
| 1 | 256 → 768 | 0.9087 |
| 1 | 256 → 256 | 0.9922 |
| 1 | 256 → 1024 | 0.9386 |
| 1 | 1024 → 256 | 1.4855 |
| 2 | 256 → 768 | 0.9444 |
| 2 | 256 → 256 | 0.8404 |
| 2 | 256 → 1024 | 0.7812 |
| 2 | 1024 → 256 | 1.1600 |
| 4 | 256 → 768 | 0.9333 |
| 4 | 256 → 256 | 1.0000 |
| 4 | 256 → 1024 | 0.9268 |
| 4 | 1024 → 256 | 1.1310 |

These isolated timings are exploratory, not full-codec or acceptance measurements. The 1024 → 256 operation was slower in all three batches. The diagnostic allocates the 4 MiB comparison buffer, so it does not measure live-pool savings. Both CPU/GPU activity and Chrome were observed in two system samples; no exclusive or quiet-GPU claim is made. The selection and initial launch order are fixed, so first-operation timing does not support a cold-start comparison.

A 2.5 MiB cap is selected after observing these results for a separate diagnostic; its heuristic could differ from the 4 MiB choice. If the only retained change were reducing the allocation, the measured prior pool plus prepared state predicts about 1.0444× at the smallest observed baseline, but actual native allocator and full-codec behavior must be measured. No production change or limit waiver follows.

[Plan](cuda-zero-workspace-poc-plan-v1.md), [source](cuda-zero-workspace-poc-v1.cpp), and [raw archive](raw/cuda-zero-workspace-poc-v1.tar.xz) (SHA-256 `125d03cce1012c07d7497f9a4e483a48f0c7003f5831dfc830fa1719473fcd42`). The archive includes all three raw reports, runner, hashes, metadata and activity samples.
