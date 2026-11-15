# Bounded CUDA workspace runtime candidate

The candidate keeps the strict-FP32 fused projection and existing model operations. On compute capability 8.x, it allocates 2.5 MiB for the backend-owned cuBLASLt workspace. The first heuristic search uses the same 4 MiB preference as the pinned wrapper; if the selected algorithm needs more than the buffer, a second search uses the buffer limit. A second oversized result or an unavailable algorithm follows the existing explicit unsupported-algorithm fallback; other errors propagate. Other compute capabilities retain the wrapper's prior 4 MiB or 32 MiB allocations. Ada devices are included in 8.x but have not been measured here.

The narrow native wrapper follows cudarc 0.19.10's layouts, transpose and bias descriptors, FP32 types, scalar values and stream-ordered pointer guards. Resource descriptors and the handle have owners; a mutex serializes submissions on the retained stream. CUDA context binding and cross-thread destruction follow the existing implementation. The [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul) specifies the workspace pointer alignment and describes the algorithm result's workspace requirement; the workspace uses the same CUDA allocation mechanism as the pinned wrapper. This is a narrowly audited draft, not a proof for every device or workload.

Focused selection tests cover fitting results, an oversized first choice, an oversized retry, and non-availability errors. The six existing codec-component tests pass on CUDA, including concurrent snapshots, cross-stream input, and an output that outlives its backend. A fresh trained CUDA capture passes **642 traces and 2,500 decisions** at existing tolerances, with zero ID disagreements and the same two near ties. No pins, model algorithms, fixtures or acceptance limits changed.

## Exploratory full-codec pool measurement

Three candidate processes used the unchanged benchmark fixture; each measured four scopes for each batch. All input/output ID hashes, output counts, and supplied-code hashes matched the earlier native baseline. These are synchronized default-pool used bytes, not whole-device allocations:

| Batch | Prior Rust bytes | Candidate bytes | Ratio |
|---:|---:|---:|---:|
| 1 | 59,677,184 | 62,324,304 | 1.044357 |
| 2 | 59,768,320 | 62,415,440 | 1.044290 |
| 4 | 59,950,592 | 62,597,712 | 1.044155 |

All four scopes and three processes produced the same values per batch. This is below the 1.05 band for this native-pool comparison, while different Torch allocator coverage and non-pool allocations remain outside the observation. The diagnostic is not the controlled acceptance run.

Compared with three earlier candidate processes in a different session, the new-to-earlier ratio of median process median wall times for batch-one reconstruction was 1.111; other measured scope ratios ranged 0.994 to 1.046. This comparison is unpaired and cannot attribute the difference to the code. Chrome and other workstation processes were sampled, so the controlled gate must resolve the timing result.

Two full-check attempts failed in unrelated evidence-tool tests: the first under umask `022` failed three tests requiring private directories; those tests pass under `077`. The second under `077` had one synthetic process-lifecycle failure, and that test passed in isolation. Both logs are retained; the isolated full rerun passed `tools/check.sh --cuda` under `umask 077`, including 484 evidence-tool tests and 18 CUDA operator cases.

[Diagnostic plan](cuda-bounded-runtime-diagnostic-plan-v1.md) and [raw evidence](raw/cuda-bounded-runtime-diagnostic-v1.tar.xz) (SHA-256 `ad2e5d3a6c8e9ca8638b8311badf0f1eac17fdadcefa489cac33361f8458d736`) include the three full-codec reports, source and binary hashes, activity, the qualification, and both failed and final passing full-check logs. The large trained trace remains local under `reports/phase2-codec/cuda-workspace-candidate-v1`. Linux controlled performance acceptance and Apple validation remain pending.
