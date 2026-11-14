# Bounded MLX memory observation for projection candidate

This diagnostic-only branch adds the previously reviewed finite memory probe
to candidate `a323b16`. After the Apple timing gate, run three model
load/drop cycles and 128 batch-4 reconstructions per retained model, sampling
MLX active/cache/peak and host RSS/HWM at fixed intervals. Input and output ID
hashes are checked against the existing qualified workload. The probe clears
cache only once before the observation; no subsequent cache reset masks growth.

Linux `cargo fmt --all -- --check` and `RUSTC_WRAPPER= cargo check --workspace
--locked` passed. The example has not been compiled or run for this runtime on
Metal. This is a bounded observation, not a timing gate or a proof of an
infinite memory bound. The previous runtime's results remain in the
[prior report](mlx-long-lived-memory-v1.md).
