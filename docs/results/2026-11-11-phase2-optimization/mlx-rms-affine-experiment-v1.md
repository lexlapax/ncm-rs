# MLX RMS weight fusion experiment

This separate branch passes a matching one-dimensional RMS weight to the pinned
MLX fast RMS normalization primitive, rather than multiplying afterward.
Other weight shapes retain the existing broadcast path. It represents the same
upstream normalization and multiplication, but rounding may differ. A Metal
regression compares the optimized path with the original broadcast composition
and checks invalid epsilon and dtype. It has not yet run on Metal.

The isolated source passes `tools/check.sh --cuda` under umask 077 with
`RUSTC_WRAPPER=`, including 18 real-CUDA operator cases. The CUDA codec bench,
trace, lifecycle, tokenizer bench and probe binaries are byte-identical to the
Linux candidate that passed 78/78. [Check log](raw/mlx-rms-affine-linux-check-v1.tar.xz)
SHA-256: `70d15d86d7a202384f6bbbedcb440566a5e235f8edd88e98b2b8d260f9fb2c03`.
No new Linux timing result is claimed for identical binaries.

The Apple initial gate for the previous runtime passed 78/78 but the fixed
control was 76/78. The proposal is not accepted; after checking Mac activity,
run Metal correctness, unchanged trained qualification and a bounded matched
diagnostic before considering it for a full gate. Preserve the prior outcomes
and all pinned tolerances; do not assert this reduces p95 variability.
