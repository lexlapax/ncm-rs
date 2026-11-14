# Experimental MLX projection plus bias

This branch is an experiment, not an accepted production candidate. For a
rank-two projection weight and matching one-dimensional bias it uses the pinned
MLX matrix-multiply-and-add primitive, preserving the upstream matrix
projection plus bias calculation. Other bias/weight shapes retain the existing
composition. Floating-point evaluation may differ, and neither correctness on
Metal nor a reduction in p95 variability has yet been established. A direct
comparison test is included for real Metal.

The isolated candidate passes `tools/check.sh --cuda` with `umask 077`,
`RUSTC_WRAPPER=` and an isolated target, including 18 real-CUDA operator
cases. [The check log](raw/mlx-addmm-linux-check-v1.tar.xz) has SHA-256
`bc89eff7ea9f96cd6ae2934850d1986249eb7074eca853b2b7175b4e38d0ba12`.
Its CUDA codec bench, trace, lifecycle, tokenizer bench and probe executables
are byte-identical to the qualified Linux 78/78 candidate. No new timing result
is claimed for those unchanged executables.

The prior Apple 75/78 result remains open. Two of nine candidate processes had
pronounced batch-4 block tail spikes, but the available data do not identify
the cause; fewer MLX operations are a hypothesis rather than a proven fix.
After Mac availability is confirmed, test this branch separately, qualify all
642 trained traces against the pinned oracle, and measure a small matched
performance diagnostic before considering promotion. Preserve the upstream
algorithm, pins, and existing tolerances.
