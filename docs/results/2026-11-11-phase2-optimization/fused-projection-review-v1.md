# Strict-FP32 fused projection: isolated patch for review

**Revised patch ready for review:** the original prototype used a Cargo target
directory shared with the main checkout, so its exact binary/source binding
remains unproven. A revised patch adds an FP32 output byte-size overflow guard
and binds the owning CUDA context on cross-thread handle drop. It was rebuilt
from scratch in a separate target directory; compiler dependency files include
the new module and backend trait source. The [revised patch](fused-projection-review-v2.patch)
applies to a clean export and matches all eight tested files; SHA-256
`a4c851cf8cf2c4b20b5c85a60ec980a0a293f5af2e7691da88d4ff8cd800741e`.

The separate-target build passes the full `tools/check.sh --cuda`, fresh 642
trained traces / 2,500 decisions with zero ID disagreements and two near
ties, and the ten-cycle, 144-request memory diagnostic with zero live CUDA pool
bytes after each model drop. It includes tests for output-byte overflow,
cross-thread backend drop, retained outputs and concurrent calls. Three matched
processes per binary show resident-block median ratios **0.76214 / 0.81971 /
0.79369** at B1/B2/B4, and decode ratios 0.97488 / 0.97116 / 0.96787; output
IDs match and maximum recorded first-logit absolute difference is 7.63e-6.
These are diagnostics, not acceptance. Pool-used peak rises by 4,194,232 bytes;
reserved peak remains 100,663,296 bytes. Baseline host HWM is 604.3–606.4 MB,
candidate 604.3–606.6 MB. [Independent evidence](raw/fused-projection-review-v2.tar.xz)
SHA-256 `d4257d10079aa2780dac9ebc29beedd6d8420c63fc787dd3cca20fc835aff702`.

The remainder records the earlier **v1** review and its provisional results;
the revised v2 patch supersedes v1 for any integration decision. Neither has
passed a full performance gate. The workspace/API risks described below remain.

**Review requested; not integrated.** The current TinyStories candidate still
fails Linux acceptance at 75/78, despite its own decode improvement. This
isolated patch targets the remaining CUDA decoder-block overhead by fusing a
projection and its bias into a cuBLASLt epilogue. The [exact patch](fused-projection-review-v1.patch)
applies cleanly to `ede7fcf` and matches the eight tested source files; its
SHA-256 is `da1ea86c2d103ce2f961a31a2c7ae3964f8956f0294d8c998c6add89f6af6e36`.
It adds a default `CodecBackend::project_bias` method whose default retains the
existing operations, and uses the fused path only for supported CUDA FP32
shapes. Other backends retain their existing operations. An unavailable fused
algorithm uses the existing CUDA projection plus addition; other CUDA errors
remain errors. The existing reduced-precision setting, non-vector broadcast
biases and empty shapes use the original path. No precision flag or tolerance
is changed.

The fused path owns a lazily created cuBLASLt handle and workspace per backend
instance, serializes host access, retains the stream and the existing cudarc
cross-stream event tracking, checks layouts, shapes and buffer bounds, and
limits unsafe code to the documented handle, allocation and matmul operations.
The pinned wrapper's default `f32` path uses TF32, so the patch explicitly
selects `CUBLAS_COMPUTE_32F`; an initial TF32 microbenchmark was rejected.
The pinned workspace helper allocates 4 MiB on this Ampere GPU and 32 MiB on
compute-capability 9 or later. The latter configuration has not been measured.
Separate backend instances can each allocate a workspace, and handle teardown
cannot report an error through Rust's `Drop`; these ownership and failure
semantics need review alongside the new backend trait method.

A predeclared, matched diagnostic compared three fresh processes per binary,
five warmups and 30 trials in every codec scope. Ratios of candidate to prior
process medians for resident decoder block were **0.83904 / 0.82345 / 0.79289**
for batches 1 / 2 / 4. Decode ratios were 0.97568 / 0.97664 / 0.97163;
encode ratios were 0.99782 / 1.00005 / 0.99855. Output IDs and input identity
fields matched in all 12 scopes; maximum recorded first-logit absolute difference
was 7.63e-6. These are diagnostic results, not acceptance. Peak CUDA pool-used
bytes rose from 72,740,536 to 76,934,768; reserved peak remained 100,663,296.
The measured host HWM ranges were 606.4–606.5 MB baseline and 604.2–604.4 MB
candidate. Allocator counters and host HWM have different coverage.

The exact patch passes a fresh trained CUDA qualification: **642 traces and
2,500 decisions**, zero ID disagreements, two near ties. Focused CPU/CUDA tests
cover broadcast fallback, empty shapes, concurrent calls and outputs
that outlive the backend. The repeated-load diagnostic passes ten model cycles
and 144 requests, with live CUDA pool use returning to zero after each model
drop while an independent backend clone remains alive. Exact-patch CUDA Clippy
and focused tests pass. A full `tools/check.sh --cuda` passed immediately before
the final change, which only narrowed `unsafe_code` lint allowances and added
safety comments; the exact patch needs the full check again after integration.
The first full-check attempt failed on a test-only Clippy warning; another failed
when an experimental benchmark inventory edit changed a hash-locked harness.
The warning was fixed and the harness restored; both failed logs are retained.

[Raw reports, plans, tests, trained qualification, resource observations and logs](raw/fused-projection-review-v1.tar.xz)
have SHA-256 `ed54061745a0fa2f86c3a46a0387daa7c67b3aa6d9303a8cbc74dc262fdcd860`.
The benchmark's existing source inventory does not list the new module or the
backend trait file; a separate eight-file source manifest binds those files and
matches a clean export with the patch applied. The raw trained trace remains
local. Linux and Apple acceptance both remain open. If this API change is
approved and integrated, repeat full Linux checks and acceptance first, then
resume the Apple optimization/gate. No benchmark or policy limit is waived.
