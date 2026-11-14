# MLX RMS weight fusion — deferred

An isolated branch passed a matching RMS weight directly to the pinned MLX
fast RMS normalization primitive; other shapes used the prior broadcast path.
This preserves the intended upstream normalization and multiplication but may
change FP32 evaluation. The branch is **not integrated** because the small
performance diagnostic did not establish a target improvement.

The isolated Linux `tools/check.sh --cuda` passed with 18 real-CUDA operator
cases; five codec/tokenizer binaries were byte-identical to the qualified
Linux 78/78 candidate. The focused weighted-versus-broadcast RMS regression,
full `tools/check.sh --mlx`, and 18 real-Metal operator cases passed. Fresh
trained Metal qualification passed 642 traces and 2,500 decisions, with zero
ID disagreements and two near ties.

Three alternating fresh process pairs gave batch-4 resident-block median
ratios **1.0673 / 0.9842 / 1.0176** (candidate/current), with a
median-of-process-medians ratio **1.0055**. Batch-2 block ratio was 1.1274;
other scopes varied widely, including unchanged encode, so the small sample
cannot establish a general regression or its cause. All paired output IDs
matched. Thirty-six activity samples showed no Chrome; WindowServer median/peak
CPU was 0.0%/3.7%. This diagnostic is not an acceptance gate and does not
replace the current Apple controlled 76/78 result.

[Metal reports, check log, qualification, activity and trace hashes](raw/mlx-rms-affine-diagnostic-v1.tar.xz)
have SHA-256 `110be156736bc14316c907f446046b1a8bc3dbb64755c4b725aec6c98363a829`.
[Linux check](raw/mlx-rms-affine-linux-check-v1.tar.xz) SHA-256:
`70d15d86d7a202384f6bbbedcb440566a5e235f8edd88e98b2b8d260f9fb2c03`.
[Exact experimental patch](mlx-rms-affine-experiment-v1.patch) SHA-256:
`279d228cdf891d597771cc176559bf2c5b55ea56bf7a6f0e8dcd92ba09e7674c`.
All six report source inventories match their respective source trees; full
trace files remain on the Mac and in the local temporary capture.
