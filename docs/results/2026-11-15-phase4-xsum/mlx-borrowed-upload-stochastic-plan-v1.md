# XSum borrowed upload: Metal stochastic output replay plan

The candidate's full top-1 Metal output matched the retained Metal baseline, but
that does not qualify its stochastic output or waive the baseline's exact
Python raw-ID failure. Reuse the already frozen stochastic follow-up protocol:
rows 512–2559, seeds 4–8, 64 contiguous batches of 32 per seed, and the same
per-batch seed `seed * 1000 + batch`. This is a replay against previously
inspected data and is not a new held-out statistical study.

Capture the qualified candidate on the clean reviewed Mac checkout, with explicit
binary, protocol, source and input identities. Preserve each batch's codes and
IDs in checked receipts and permit resumption only if the recorded identities and
existing receipts match. For each seed compare every candidate code and raw ID
against the previously scored Metal capture. An exact match allows reuse of that
deterministic scorer's input and retained comparison; it is not a fresh scorer
run and does not erase the original CUDA stochastic failure or the Metal top-1
raw-ID failure. If any output differs, retain the result and investigate or
rescore under the unchanged frozen protocol; do not regenerate a reference or
change a quality margin.

Linux focused tests passed for seed/batch receipts, resume ordering and exact row
attribution. The complete comparator also replayed all five retained scored
Metal captures against themselves: 10,240 rows with zero differences. That
self-comparison SHA-256 is
`70e56b06063a483a9b60de992bf28135fd2292ab9818cede1c89fc375b2ddcbf`.
Publish the tools without changing the qualified XSum runtime source, then run
a one-batch Metal smoke and unchanged resume before capturing all five seeds.

On clean Mac commit `a10f358935ec13f43f6adfad54d494ecc868c103`, the seed-4
one-batch smoke matched the retained Metal codes and IDs, and an unchanged
resume passed. Its manifest SHA-256 is
`136564f8719fe9fcd4bbf77b9aa1afa719051a61d2e05dbbf257b10530f05902`.
The five full seeds are running sequentially using
`.reference/mlx-venv/bin/python tools/bench/capture_xsum_mlx_stochastic_candidate.py
--head a10f358935ec13f43f6adfad54d494ecc868c103 --binary-sha256
a580ba2ca6c8a2e1a43d4e7a0b3a7b6645b6576f7424ec2ea3adb711a2429414
--qualification reports/phase4-xsum-borrowed-upload-mlx-cli-v3/result.json
--qualification-sha256 30415fa65a9346989ea8b4924a27d565bfd448ece82c310cb09adc148d7f3d12
--seed SEED --output reports/phase4-xsum-borrowed-upload-mlx-stochastic-seed-SEED-v1`
for seeds 4 through 8. Partial captures are not a quality result.
