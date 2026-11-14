# Decoder normalization candidate Apple gate — 77/78

Candidate `884d8f6` is runtime-equivalent to Linux-tested `621c565` and passes
the complete MLX check, including the new direct normalization regression and
18 real-Metal operator cases. All 642 trained traces pass against the pinned
reference with zero ID disagreements; four tokenizer probes pass. A fresh
45-process three-way matrix and separate nine-process lifecycle capture used
the unchanged accepted Rust `3307dff`, qualified Python, workloads and limits.

The initial audit passed **77/78**. GPT-2 encode met the point limits against
accepted Rust (median ratio 0.97365, p95 ratio 0.99344) but its process-median
envelope was 1.09134 against the 1.05 limit. Before collection, a single
bounded control was declared for all GPT-2 scopes and checked load: nine
fresh processes per runtime. The controlled audit remains **77/78**: GPT-2
encode median ratio 0.98616, p95 0.99921, with paired-process 95% median
interval **[0.93033, 1.06574]** against the 1.05 upper limit. The control plan
SHA-256 is `c9caccae1e9e949adcb4b2acb35d2508c9494ac46ff75a3925f0a368e66ebc33`.

The earlier block regression is resolved in this matrix. Resident-block
median ratios versus accepted Rust at batches 1/2/4 are **0.82743/0.77710/
1.02875**, and reconstruction ratios are **0.26673/0.27457/0.36610**. This
is a substantial retained improvement, but the GPT-2 uncertainty keeps
the overall Apple gate and Phase 2 open. No acceptance limits or system clock
or power settings changed.

[Raw matrix, control, quality, lifecycle, telemetry, plans and audits](raw/blocknorm-apple-gate-v1.tar.xz)
have SHA-256 `155ff6ab0dbc341a7e43dd8aa0ee5e02363066226ad3422745cfc142e106d4ee`.
The controlled audit SHA-256 is
`361d0ee4d0e56f84f19e5f58ac0b7a110510d78b71c503f62bf3895831cf56cf`;
it reproduces byte-for-byte from a clean `884d8f6` export and the archive,
including the same failure. Activity samples do not establish exclusive GPU
access.
