# MLX fast normalization — isolated experiment deferred

A temporary candidate replaced the composed layer and RMS normalizations with
pinned MLX fast primitives, with the existing epsilon validation retained.
Layer normalization is also used by group normalization. No production code
changed. The candidate passed existing real-Metal operator/component tests and
all 642 trained traces / 2,500 decisions against the unchanged oracle.

Three matched fresh processes per native binary compared the candidate with
the committed linear-gather implementation, using the existing codec fixture,
five warmups and 30 trials. Median-of-process resident-block ratios at batches
1/2/4 were **0.818/0.751/0.816**, but encode ratios were **0.986/1.084/1.093**
and reconstruction ratios **0.967/0.951/1.088**. Thus the small sample shows
material regressions in other scopes despite the block improvement. MLX peak
counters were slightly lower in the sampled fast-normalization processes; the
observed host HWM was similar. These measurements are diagnostic, not a final
acceptance test, and do not justify promoting the mixed candidate. No adaptive
retry was run.

[Source patch, validation, trained qualification and raw matched reports](raw/mlx-fastnorm-poc-v1.tar.xz)
have SHA-256 `119fb4ba5077701779807aa30698298c0f62689068139f087165e649088b804f`.
