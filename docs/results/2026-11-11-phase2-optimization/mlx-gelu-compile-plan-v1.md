# Exact GELU compilation diagnostic

The current decoder uses exact-erf GELU on a 1,024-wide hidden layer. Test the
same MLX operation order in an isolated example with all arrays, including
constants, passed explicitly to compilation. This examines a possible reduction
in per-block GPU work; it does not assume compilation fixes the observed
encoding-to-GPU delay.

Use three fresh processes with shapes B1/B2/B4 × 89 × 1024, five warmups and 30
alternating-order paired evaluations per shape. Compare every output element
against the uncompiled expression using the existing primitive tolerance.
Record first compiled graph/total time and MLX peak/cache counters as well as
steady timings. Retain all processes and failures; do not change production,
pins, precision, acceptance limits or goldens. The microbenchmark is not
trained-model parity, full-model performance or an acceptance gate. Integrate
nothing unless any benefit survives startup/cache, semantic and full-codec
review on both platforms.
