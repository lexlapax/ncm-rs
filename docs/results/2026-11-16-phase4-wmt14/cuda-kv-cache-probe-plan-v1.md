# WMT14 upstream KV-cache diagnostic

The [synchronized stage profile](cuda-stage-profile-v1.md) shows that
repeated transformer work dominates the measured Rust intervals. The pinned
upstream implementation exposes `use_kv_cache=True`, but its cache path omits
the prefix padding bias both when populating cached prefix keys/values and in
the later level mask. The released WMT14 prefixes contain padding, so this
path cannot be assumed semantically equivalent to the qualified no-cache
reference. No runtime switch or reference change is implied.

A separate, pinned-source Python diagnostic will run the unmodified upstream
cache option on the first 32 prefixes, with the same FP32, math-SDPA and timing
scope as the existing benchmark. It will compare every code and raw ID to the
retained no-cache Python capture and preserve failures, timing and source/input
hashes. Even a match on this small sample does not qualify the path or prove
padding correctness. If it differs, those differences are evidence against
using the upstream cache path unchanged. Linux CUDA runs precede any Metal work.
