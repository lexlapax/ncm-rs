# WMT14 CUDA performance comparison plan

Compare the pinned upstream Python reference and experimental scoped
reference-order CUDA backend on the same first WMT14 prefixes, in FP32, for
released CFG=0 and diagnostic CFG=1, at batches 1, 4 and 16. Each workload
uses three fresh processes per runtime in alternating order, five warmups and
30 synchronized trials per process. Timed work includes prefix upload,
generation, codec decoding, greedy selection and codes/raw-ID host readback;
model loading and explicit Python garbage collection are outside that interval.
Report Python cleanup separately and include a cleanup-inclusive comparison.
CUDA events also bracket upload through greedy selection, excluding the final
codes/IDs readback. Because the Rust generator performs intermediate host
readbacks, event elapsed time can include stream idle while the host works; it
is not a sum of kernel execution times.

The supervisor verifies matching output codes and raw IDs between every trial
and process, records process memory and roughly one-second GPU telemetry,
fingerprints the committed harness, artifact, inputs, binary and cuDNN
libraries, and refuses dirty tracked source or changed inputs. Report median,
p95, dispersion, throughput, host RSS and sampled GPU memory with sampling and
allocator limitations. Keep failures and OOMs visible. GPU desktop activity
will be reported; no exclusivity is assumed. This performance experiment is
not another independent quality study and does not resolve Metal correctness.

An initial batch-1 CFG=0 harness smoke matched outputs, with 30 trials after
five warmups in each runtime; it is not a performance acceptance result. The
Python probe was subsequently adjusted to release an additional temporary
before cleanup, so the formal committed harness must be rerun.

The local `setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda`
passed all requested checks and 18 CUDA operator cases (log SHA-256
`91db06e51b43d7ddf0161f04491df856de51610128abf9edd38c44d7c1ef74b4`).
The WMT experimental/cuDNN example also passed feature-specific Clippy with
`-D warnings` and built in release mode.

A separate CFG=0 batch-32 capacity comparison is prepared with the same
alternating three-process, 5-warmup/30-trial protocol. The scoped CUDA
reference normalization accepts an internal batch at most 32; CFG=1 doubles
its internal batch, so its supported maximum external batch is 16. The
supervisor and both probes reject CFG=1 batch 32 explicitly. This is a supported
backend boundary, not a claim about the GPU's physical maximum batch. The
batch-32 example passed feature-specific strict Clippy and release build;
the supervisor rejects the unsupported combination. Formal capacity results
remain pending and will retain any failures or OOMs.
