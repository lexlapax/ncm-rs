# Experimental MLX decoder normalization

This branch is an experiment, not an accepted production candidate. On the
existing accelerated decoder normalization path, use the pinned MLX fast
layer-normalization primitive's weight and optional bias arguments when they
are one-dimensional and match the final input dimension. Other shapes retain
the existing separate multiply/add path. This preserves the intended upstream
normalization, scale and bias operation, but changes floating-point evaluation;
Metal numerical parity and performance must be measured before any promotion.
A direct weighted/bias regression is included and requires real Metal.

On Linux the isolated source export passes `tools/check.sh --cuda` with
`umask 077`, `RUSTC_WRAPPER=` and an isolated target: 18 real-CUDA operator
cases passed. The initial sandboxed attempt failed before compilation because
`sccache` lacked permission; both logs are in
[the archive](raw/mlx-affine-linux-check-v1.tar.xz), SHA-256
`698431579594c3d1a54103433e0d858b87188d0bbed569f40d63173d533054bc`.
The release CUDA codec bench, trace and lifecycle and tokenizer bench and probe
are byte-identical to the previously qualified Linux candidate. Thus the Linux
78/78 evidence applies to these identical measured binaries; no new timing
result is claimed. The changed MLX path has not yet run on Metal.

Next: inspect Mac availability, test this reviewed branch in a separate Mac
worktree, establish Metal correctness including the 642 trained traces, and
measure a small matched diagnostic before deciding whether to integrate. The
current Apple 72/78 gate remains open; no limits, pins or upstream algorithms
are changed or waived.
