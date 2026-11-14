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
activity and output hashes. The prepared [runner](mlx-attention-nomask-run-v1.py) records the paired
processes and activity. This is diagnostic, not acceptance. Integrate
nothing or run a full gate without a defensible performance and correctness
result; retain negative results and the previous failed Apple controls.

The isolated Linux checkout passed `tools/check.sh --cuda`, including all 18
real-GPU operator cases. Its target began as a copy of the earlier target;
all local workspace packages were cleaned and rebuilt. The codec benchmark,
trace and lifecycle executables are byte-identical to the qualified Linux
candidate: respectively
`925363bce9e04204f27f4035fdc9a305aac40cfd77f5a02d5baffcfb860e20e0`,
`7dc4cb57ef9a3d0e0eabd496ccdeb35d6fbb1e579cda39278099862ae1701c85`,
and `be60ca9b10ac0d5a3063b0548a25b8b978ac5775183611986ca923d5d14c80ed`.
The benchmark depfile names this new worktree, not the earlier one, and omits
the platform-inactive MLX source. The [check log and depfile](raw/mlx-attention-nomask-linux-v1.tar.xz)
have SHA-256 `60c12c762b9bec6662e65786cad8361d71845b162041472937b0ba24bece1b7a`.
The existing Linux trained qualification and 78/78 gate remain bound by those
identities. Metal correctness and performance are still pending.
