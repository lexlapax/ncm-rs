# MLX projection-plus-bias candidate

The candidate passes the existing projection and matching vector bias to the
pinned MLX matrix-multiply-and-add primitive. Other bias or weight shapes keep
the prior composition. The upstream projection-plus-bias operation, artifacts,
pins and numerical tolerances remain unchanged; evaluation order can differ.
The direct Metal regression compares the fused and separate computations and
covers batched and vector inputs, broadcast fallback, invalid width and dtype.

The isolated source passes `tools/check.sh --cuda` on Linux under umask 077 and
`RUSTC_WRAPPER=`, including 18 real-CUDA operator cases. CUDA codec benchmark,
trace and lifecycle and tokenizer benchmark and probe binaries match the
qualified 78/78 Linux candidate byte-for-byte. [Linux check log](raw/mlx-addmm-linux-check-v1.tar.xz)
SHA-256: `bc89eff7ea9f96cd6ae2934850d1986249eb7074eca853b2b7175b4e38d0ba12`.
The full `tools/check.sh --mlx`, focused Metal regression and 18 real-Metal
operator cases pass. Fresh trained qualification passes 642 traces and 2,500
selection decisions, with zero disagreements and two near ties.

Three alternating fresh pairs of full codec diagnostics show batch-4 resident
block median ratios of **0.9712, 0.9693, 0.9228** (candidate/current), with
median-of-process medians ratio **0.9693**. Their individual p95s in ms were
1.25637/1.36987, 1.27025/1.20629 and 1.14188/1.20608 (candidate/current).
All paired output ID hashes matched; unrelated batch-4 encode varied, with a
median ratio of 1.1231. The diagnostic does not establish a p95 improvement,
a causal explanation for the earlier spikes, or acceptance. In 36 activity
samples Chrome did not appear; WindowServer median/peak CPU was 17.25%/47.8%.
No thermal warning was reported before or after collection.

[Metal diagnostics, logs, qualification and trace hashes](raw/mlx-addmm-metal-diagnostic-v1.tar.xz)
have SHA-256 `c71967ec6417f139ea33c753dce4e2ba9f2eeafd7e8b236327eec1f76a55139b`.
All six report source inventories match their source trees; trace files remain
on the Mac and in the local temporary capture. The unchanged acceptance limits
require a fresh complete Apple gate. The prior 75/78 control remains retained.
