# Isolated fused Metal interpolation diagnostic

The existing gather interpolation accelerated B4 encode while increasing peak
MLX active allocation by about 26 MB in matched full-model runs. This diagnostic
asks whether a single Metal operation can preserve its host-computed f64
coordinates and separately rounded f32 weights while avoiding the two gathered
intermediate tensors. It is exploratory and does not change production code.

The first isolated Mac build stopped at the workspace unsafe-code lint; no GPU
code ran. The narrowly scoped allowance on the audited kernel call follows the
existing project pattern. Preserve the failed build log alongside the follow-up.

Copy the [diagnostic source](mlx-linear-fused-poc-v1.rs) verbatim into an
isolated checkout as `crates/ncm-backend-mlx/examples/mlx_linear_fused_poc.rs`.
Build a locked, offline release example against the existing pinned native MLX
and capture the build log, source, dependency file, binary hash and environment.
The current unchanged production candidate already passed the required Linux
CUDA gate; this Metal-only diagnostic does not alter the shared model or CUDA.

In a quiet Mac window run three fresh processes each of `gather` and `fused`,
in rotating order; use unique new report paths and retain every outcome. Each
process tests ten BLC shapes, including the qualified model's 256 channels and
B4, and verifies every finite output against the current production gather
operator before measuring it. Record output bit differences, first evaluation
before in-process parity, 30 post-warmup elapsed samples, and peak, active and
cached MLX allocator counters. Observe desktop activity and thermal status. Do
not clear any system cache; the diagnostic clears only its own process-local
MLX buffer cache between cases. Preserve failures and stop on a failed process.

The kernel uses host-computed indices and weights exactly as the current path;
its bounded launch writes one output per thread and requests row-contiguous
inputs. Floating-point contraction is disabled in the source. This experiment
only checks synthetic finite contiguous arrays. It does not establish all-shape,
nonfinite, noncontiguous, trained-model, cold-cache, lifetime, or safety parity.
First-use compile latency and thread-safety/caching integration are explicitly
unresolved. MLX allocator counters are not device-wide memory or host RSS.
Any useful result needs separately scoped validation before production adoption;
no tolerances, accepted artifacts, or gate criteria are changed.
