# Strict-FP32 fused projection: isolated patch for review

**Provenance recheck in progress:** the prototype was built using a Cargo target
directory shared with the main checkout. Subsequent inspection found that the
two checkouts can reuse the same artifact paths, so the claimed exact-source
binding for the prototype binaries and measurements is not yet established.
A clean build in a separate target directory is underway. Treat the performance
and qualification results below as provisional until that recheck completes;
they do not authorize integration or acceptance.

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
