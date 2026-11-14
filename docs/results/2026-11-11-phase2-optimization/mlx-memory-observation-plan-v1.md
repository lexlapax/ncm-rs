# Bounded Apple memory observation — prepared, not run

This diagnostic-only branch prepares the remaining Step 4 resource evidence.
With the pinned trained artifact, it will load and drop three separate Metal
models and perform 128 batch-4 reconstructions per model. It records MLX active,
cache and peak allocation counters and host RSS/HWM after load, after requests
1, 8, 32, 64 and 128, and after every model drop. Each request's combined code
and output-ID digest must match. A single initial cache clear precedes all
cycles; no per-request or per-cycle cache clear obscures observed growth.

This is a fixed finite observation, not a timing benchmark or proof of an
infinite bound, and MLX allocator counters are not separate VRAM measurements.
The code and fixture follow the existing codec benchmark's deterministic batch
construction. `cargo fmt --all -- --check` and `RUSTC_WRAPPER= cargo check
--workspace --locked` pass on Linux; the Metal example is excluded there and
has **not yet been compiled or run on Apple hardware**. The user's Mac reboot
must finish before reconnecting. After readiness, use a reviewed pushed commit
in a separate Mac worktree and retain the report and source/binary bindings.
