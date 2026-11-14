# Combined candidate Apple initial gate — 69/78

The current source-equivalent candidate `2df27c3` passed all 642 fresh trained
Metal traces and 2,500 decisions against the unchanged oracle, with no ID
disagreements, and all four canonical tokenizer profiles. The exact code had
already passed `tools/check.sh --mlx` in the isolated BERT review worktree; the
commits from `4ab4d3a` to `2df27c3` change documentation and evidence only.
The accepted Rust comparator is `3307dff`; the pinned adapted Python comparator,
workloads and limits are unchanged. A fresh 45-process matrix and separate
nine-process lifecycle capture completed.

The initial audit passes **69/78**, not the final gate. Six misses are resident
decoder-block comparisons: candidate/accepted Rust median ratios for batches
1/2/4 are **1.05010/1.08786/1.35310**, and all three also miss against Python.
Qwen encode and both TinyStories decode modes additionally fail the three-process
uncertainty envelope against accepted Rust. BERT encode passes its initial
comparison at median ratio 0.93718 and p95 ratio 0.95277. Reconstruction median
ratios versus accepted Rust at batches 1/2/4 are 0.27043/0.28012/0.39638.
All limits remain unchanged. The process-activity sampler recorded 108 samples;
the existing `rmlx` process did not appear among the busiest sampled processes,
which does not prove exclusive GPU access or explain the regression.

The initial audit SHA-256 is
`81d01768a59a4eb3aa598e1d9ec9dc3584fca4def3c60160aa08d2f1cbecb0e6`; it
reproduces byte-for-byte with the archived reports in a clean `2df27c3` source
export. The source inventories of all 75 report files, including shared Python
copies, match the candidate or clean `3307dff` baseline export. This confirms
provenance, not acceptance. [Raw matrix, quality, lifecycle, tokenizer probe,
telemetry and audit](raw/final-mlx-initial-v1.tar.xz) SHA-256:
`e25c768c8182b169306f7c306ed1b13aac19456baf56f3261b369803277f332f`.
No final control has been run for this candidate; investigate the observed
block regression rather than silently narrowing or retrying the gate.
