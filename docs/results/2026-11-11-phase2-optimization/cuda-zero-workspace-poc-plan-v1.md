# Zero-workspace fused projection diagnostic

The retained Linux gate reports synchronized candidate CUDA default-pool used
bytes 1.07039–1.07071× accepted Rust, with a 4 MiB backend workspace and 25,680
bytes of prepared state matching the observed increase. The current 78-cell
audit tests host RSS/HWM and does not decide this separate live-memory issue.
The user approved the existing projection patch with its workspace cost
documented; that did not waive the 1.05 live-memory band.

NVIDIA documents that the `cublasLtMatmulPreference` maximum workspace defaults
to zero and may constrain heuristic selection. The pinned `cudarc` 0.19.10
`Workspace` constructor instead reserves 4 MiB on Ampere and does not expose a
public size-selecting constructor. Do not change or replace the dependency in
this experiment. Compare a direct cuBLASLt zero-workspace request against the
existing 4 MiB preference in a standalone CUDA diagnostic, using strict FP32,
identical inputs, output layouts and bias epilogue. The diagnostic is not the
production Rust implementation and does not establish safe Rust FFI integration.

Use the four qualified decoder projection dimensions, B1/B2/B4 × 89 positions,
for 12 cases. Each of three fresh processes compares both modes, alternates their
order across trials, checks every output against the 4 MiB result within the
unchanged component tolerance, and separately checks selected elements against
host arithmetic. Record all 30 timed CUDA-event samples after five warmups, first
calls, algorithm availability and workspace returned by the heuristic. Capture
GPU/desktop activity, hashes, source, compiler/toolkit versions and failures.
Retain unsupported shapes and negative results; do not silently fall back or
retry. The 4 MiB buffer may be allocated in the diagnostic process for the
comparison, so no live-pool reduction is claimed from this isolated test.

NVIDIA [cuBLASLt preference documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulpreferenceattributes-t)
defines the maximum workspace preference and its zero default. Any resulting
candidate still needs audited ownership, concurrency and error handling, CUDA
qualification and full Linux performance/live-memory validation before Apple;
no tolerance, artifact or production source changes are authorized by a
successful microbenchmark alone.

The prepared standalone source compiles locally with
`/opt/cuda/bin/nvcc -std=c++17 -O2 -ccbin /usr/bin/g++-15 -x cu cuda-zero-workspace-poc-v1.cpp -lcublasLt -o /tmp/ncm-cuda-zero-workspace-poc-v1`
(the source path is relative to this directory). The source SHA-256 is
`cb0abd357279451c5e8cc39b2426da1263c86a4ba39fae0939b766f30cf6168c`;
the initial executable SHA-256 is
`44f55fbaac6343cc8435b36a72ec930e44656911fbdd90329110912c29b6eb80`.
The grouped run is not yet evidence until its individual results are retained.
