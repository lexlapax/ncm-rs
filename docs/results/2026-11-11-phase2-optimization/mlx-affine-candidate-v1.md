# MLX decoder affine-normalization candidate

The candidate passes matching one-dimensional weight and optional bias to the
pinned MLX fast layer-normalization primitive on the existing accelerated
decoder path. Other shapes keep the prior separate operations. It expresses
the same normalization, scale and bias as the upstream decoder, but may
round differently. The pinned model, oracle, precision and tolerances remain
unchanged.

On Linux, `tools/check.sh --cuda` passed in an isolated clean source export
with `umask 077` and `RUSTC_WRAPPER=`; 18 real-CUDA operator cases passed.
The first sandboxed attempt failed before compilation when `sccache` lacked
permission. [Both logs](raw/mlx-affine-linux-check-v1.tar.xz) have archive
SHA-256 `698431579594c3d1a54103433e0d858b87188d0bbed569f40d63173d533054bc`.
The CUDA codec benchmark, trace, lifecycle, tokenizer benchmark and probe
binaries are byte-identical to the qualified Linux candidate that passed
78/78; no new performance run is claimed for unchanged executables.

On Apple, the weighted normalization regression and full `tools/check.sh
--mlx` passed, including 18 real-Metal operator cases. Fresh qualification
against the pinned trained oracle passed 642 traces and 2,500 decisions, with
zero disagreements and two near ties. Three fresh paired diagnostic processes
per binary, alternating order, yielded batch-4 resident-block median ratios
0.9405, 0.9458 and 0.8081, with median-of-process medians ratio 0.9065.
The corresponding individual sampled p95s were 1.3793/1.3130,
1.4574/1.4405 and 1.3447/1.7737 ms (candidate/current), so the small
sample does not establish a p95 improvement. Other scopes vary, including
batch-2 reconstruction median ratio 1.2964 and batch-4 encode 1.1039; this is
not an acceptance result. All paired output IDs matched. The six source
inventories match their respective source trees. In 35 activity samples,
Chrome did not appear and WindowServer median/peak CPU was 21.3%/32.4%.

[Diagnostic evidence](raw/mlx-affine-metal-diagnostic-v1.tar.xz) SHA-256:
`ec7e977db66683314d11e4b71ce35d05f0dc83b3ead0f20ddc382e899dc11697`.
It contains the runner, commands, raw reports, activity, thermal observations,
check logs, qualification and trace hashes. Full traces remain in the Mac and
local temporary captures. The candidate requires the fixed complete Apple gate;
the previous 72/78 result is retained. Do not claim success from this diagnostic.
