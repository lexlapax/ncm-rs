# Shape-independent GELU: bounded lifetime observation

The shape-independent prototype improves warm resident-block medians, but its
first compiled scope has twice shown a roughly quarter-second outlier. This
isolated diagnostic exercises the existing experimental backend in separate
eager and compiled processes. Each process creates and drops three complete
backends/models; each lifetime executes 16 trained reconstructions with batches
1–5. Capture the first and every subsequent execution, matched output-ID hashes,
MLX active/cache/peak bytes and host RSS/HWM before, during and after drops.
No system or global MLX cache is cleared, so a fresh process is not a controlled
cold-system-cache measurement. The 48 requests per mode are a finite observation,
not a general memory bound or acceptance measurement. Report unfavorable results.

Validate existing qualified Linux source and executable hashes before Metal.
Use the previously reviewed shape-independent toggle patch only in a separate
checkout and verify its source identity. The diagnostic does not alter model
math, tolerance, pin or production behavior. A production version must still
resolve the pinned Rust compile constructor's panic on cache-acquisition error,
including environments built with panic-abort; it must not hide compile failures
or move expensive compilation into unreported request phases. Full trained
qualification, platform checks, lifecycle comparison and unchanged gates remain
necessary. If startup or lifetime is materially worse, do not integrate on the
strength of block medians alone.
