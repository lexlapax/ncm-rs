# TinyStories bounded decode reservation

The native byte-BPE decoder previously grew a fresh byte vector geometrically.
It now reserves at most 4,096 bytes based on the input ID count before decoding.
The cap bounds eager allocation for long or all-special requests. Token ID
validation, the 16 MiB output bound, special-token handling and malformed UTF-8
replacement remain unchanged. A regression covers long inputs, skipped special
tokens, malformed UTF-8 and an invalid final ID.

A predeclared Linux-only diagnostic interleaved three processes of each binary,
with five warmups and 30 trials across all TinyStories scopes. Ratios of the
median of candidate process medians to the baseline were 0.99075 encode,
0.79144 decode and 0.82950 skip-special decode. Every output hash matched.
This small sample is diagnostic evidence, not final acceptance or a claim for
unseen workloads. Candidate report source inventories match the candidate source.

The full `umask 077; RUSTC_WRAPPER= ./tools/check.sh --cuda` passes, including
real CUDA tests and 18 operator cases. The tokenizer suite and all four real
asset probes pass. This change does not affect codec code; its CUDA benchmark
binary remains byte-identical to the previously trained-qualified and
Linux-gated candidate. The [raw plan, reports, output hashes, checks and probes](raw/tinystories-reserve-v1.tar.xz)
have SHA-256 `b467c3b340a3ce8143cf7f82ae2a0e32567260d9a3233498efb588e17878759c`.

Before measuring on Apple, run a fresh Linux full matrix with three processes
per runtime and workload, followed unconditionally by a nine-process-per-runtime
all-workload control and a fresh nine-process lifecycle capture. Use the same
baselines, GPU policy, CPU/observer placement, metrics and acceptance limits as
the [previous gate](conv-bias-linux-gate-v1.md), retain every result and do not
select or retry based on outcomes. Apple validation and the batch-4 Metal block
remain pending.
