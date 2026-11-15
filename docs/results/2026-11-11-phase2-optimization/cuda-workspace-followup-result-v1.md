# CUDA capped-workspace follow-ups

Two exploratory comparisons each ran three fresh processes over the same 12 strict-FP32 fused projection shapes, with five warmups and 30 interleaved event timings per mode. All results passed the diagnostic's existing per-element tolerance; CPU reference samples also passed. These are synthetic operations, not full-codec benchmarks or acceptance gates. Ratios compare per-process medians to the 4 MiB baseline, then take the median of three ratios.

## Capped comparison

A 2.5 MiB heuristic preference selects different algorithms from the 4 MiB preference, with mixed GPU timings. The largest absolute output difference was 3.09944152832e-06.

| Batch | Projection | Median of process GPU-time ratios |
|---:|---|---:|
| 1 | 256 → 768 | 0.9087 |
| 1 | 256 → 256 | 1.0938 |
| 1 | 256 → 1024 | 0.9366 |
| 1 | 1024 → 256 | 1.1875 |
| 2 | 256 → 768 | 1.0000 |
| 2 | 256 → 256 | 0.8571 |
| 2 | 256 → 1024 | 0.7812 |
| 2 | 1024 → 256 | 1.1214 |
| 4 | 256 → 768 | 1.0143 |
| 4 | 256 → 256 | 1.0040 |
| 4 | 256 → 1024 | 1.0000 |
| 4 | 1024 → 256 | 1.0238 |

[Plan](cuda-capped-workspace-poc-plan-v1.md), [patch](cuda-capped-workspace-poc-v1.patch), [raw evidence](raw/cuda-capped-workspace-poc-v1.tar.xz), SHA-256 `29919e5decfffa43373c1031e911ae3b3de14b898478465e9cab4e1e182f967f`.

## Selected comparison

Using the 4 MiB heuristic preference while supplying a 2.5 MiB workspace produced numerically identical outputs to the 4 MiB baseline for every element in all cases. The largest selected workspace requirement was 2,187,264 bytes; this approach needs a general fallback when a selected algorithm needs more than the actual buffer.

| Batch | Projection | Median of process GPU-time ratios |
|---:|---|---:|
| 1 | 256 → 768 | 1.0000 |
| 1 | 256 → 256 | 0.9926 |
| 1 | 256 → 1024 | 1.0000 |
| 1 | 1024 → 256 | 1.0000 |
| 2 | 256 → 768 | 1.0000 |
| 2 | 256 → 256 | 0.9978 |
| 2 | 256 → 1024 | 1.0000 |
| 2 | 1024 → 256 | 0.9950 |
| 4 | 256 → 768 | 1.0037 |
| 4 | 256 → 256 | 1.0020 |
| 4 | 256 → 1024 | 1.0000 |
| 4 | 1024 → 256 | 1.0004 |

[Plan](cuda-selected-workspace-poc-plan-v1.md), [patch](cuda-selected-workspace-poc-v1.patch), [raw evidence](raw/cuda-selected-workspace-poc-v1.tar.xz), SHA-256 `aa59ab07f81d2d46e7830832d50babdad59f3ae99260a617890d1664a5d1380e`.

The shared diagnostic allocation remains 4 MiB; actual CUDA pool savings have not been observed. Sampled desktop activity means these exploratory timings do not establish quiet-GPU performance. Fixed setup/launch ordering precludes cold-start conclusions. cuBLASLt's pinned Rust wrapper exposes only its own fixed-size workspace, so using the selected-algorithm approach requires an audited integration with correct resource, stream, fallback and error handling. Production correctness, broader shapes, Linux gate and then Apple validation remain pending. No acceptance limits or upstream math changed.
