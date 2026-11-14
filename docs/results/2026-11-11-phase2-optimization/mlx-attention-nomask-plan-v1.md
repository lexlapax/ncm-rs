# MLX unmasked attention runtime experiment

The shared decoder requests bidirectional attention with `None` for its mask,
but the current MLX adapter constructs and passes a scalar zero mask. Passing
no mask expresses the same mathematical operation without constructing that
array. Explicit masks must remain unchanged and `None` must never become
causal. This experiment keeps the public `ops::attention` masked interface and
adds an internal optional-mask route; it does not change shared model code,
pins, precision, workloads or tolerances.

Research on pinned MLX 0.32.2 shows that an explicit mask is broadcast into
attention's inputs ([source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/fast.cpp)),
and Metal specializes its attention path according to mask presence
([source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp)).
For the qualified model's query length 89 and head width 64, the source routes
supported inference to fused full attention by default; forcing a fused path is
not the proposed change. This indicates a plausible reduction in mask-related
work, not a proven speedup or an explanation for the observed scheduling tail.
Apple documents that committed buffers are scheduled asynchronously, so a
shorter graph does not by itself establish lower queueing delay
([documentation](https://developer.apple.com/documentation/metal/setting-up-a-command-structure)).

Apply the adjacent zero-context patch only in an isolated checkout after
verifying the original SHA-256 values: `ops.rs`
`43339b8f874de5192850304d7ef96b87171bc3d1ba5d60f60883c2b45d6ab91e`,
`codec.rs` `2283893a503fbf18de4a92024971cd4cac010a2032acde1f752c14fe39bdea4f`,
and `codec_components.rs`
`886d1ecf2fba7aedd2c5c04334590db0fd2fbf3186eeafab39913fb1e934aef9`.
Use `git apply --unidiff-zero` and retain the patch and all outcomes.

Validate Linux/CUDA first and bind unchanged Linux codec binaries to the
qualified candidate. On real Metal, test no mask versus an explicit zero mask,
nonzero masks, different sequence/head dimensions and batches, then run the
full MLX check and qualify all trained traces with unchanged budgets. Only if
correctness passes, compare current and experimental full codec in three fresh
alternating process pairs, recording all batches/scopes, startup, host memory,
activity and output hashes. This is diagnostic, not acceptance. Integrate
nothing or run a full gate without a defensible performance and correctness
result; retain negative results and the previous failed Apple controls.
