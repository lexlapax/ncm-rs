# Current candidate Apple gate

Candidate `31ebb2d` (runtime `928894b`) passes `tools/check.sh --mlx`, four
cleared-environment tokenizer probes, and trained Metal qualification: 642
traces, 2,500 decisions, zero disagreements and two near ties under the pinned
oracle and existing tolerances. The fixed initial 45-process matrix passes
**69/78**; its unconditional all-workload 135-process control passes **72/78**.
The nine lifecycle runs agree on output IDs. Phase 2 remains incomplete.

The controlled audit leaves these six cells pending:

| Cell / comparison | Evidence |
| --- | --- |
| Batch-4 resident block / previous Rust | p95 interval upper 1.11787 > 1.10 |
| Batch-4 resident block / Python | p95 interval upper 1.14579 > 1.10 |
| TinyStories encode / previous Rust | p95 interval upper 1.11185 > 1.10 |
| TinyStories checked load / previous Rust | p95 interval upper 1.19986 > 1.10 |
| GPT-2 VmHWM / previous Rust | peak ratio 1.10197 > 1.05 |
| GPT-2 VmRSS / previous Rust | peak ratio 1.10197 > 1.05 |

For GPT-2, one candidate process had 27,443,200 bytes at its final snapshot,
versus the largest previous-process snapshot of 24,903,680 bytes; that candidate
already had a higher load snapshot. This does not identify the cause.

Activity was sampled every two seconds. The initial run had 107 samples, no
Chrome among the busiest processes, and WindowServer median/peak 17.1%/34.2%.
The control had 316 samples, WindowServer median/peak 18.7%/46.6%, and Chrome
or its helpers appeared 172 times across the top-process entries, with maximum
sampled process CPU 82.9%. These samples do not establish GPU exclusivity or a
cause for any cell. No thermal warning was reported before or after either run.
No clock or power policy changed.

The [raw archive](raw/fused-apple-gate-v1.tar.xz) has SHA-256
`98bde4afecdd88a1d2cf6f72b86cbe4eaf22a173f9ee234f9824623f45ecbbc0`.
It retains the predeclared plan, both matrices, activity, lifecycle, tokenizer
probe, qualification, test/build log, audit configurations and results, clean
source audit and trace-file hashes. Full trace files remain on the Mac and in
the local temporary capture; the archive contains their hashes, not their
bytes. The relative-path controlled audit from a clean candidate export has
SHA-256 `60499c0a3e0decd23ca9f515cd40745a8d9fb8446996610a202ebc98b6155e48`.
All 120 native report inventories and 180 shared Python report copies match
clean baseline and candidate exports. The compiler dependency file includes
`crates/ncm-core/src/lib.rs`, whose SHA-256 matches the clean candidate; the
built and measured codec binary hashes agree. The standard benchmark inventory
omits that trait source, so this separate binding is necessary.

Do not repeat the unchanged candidate. Investigate the batch-4, TinyStories,
and GPT-2 findings, retain this outcome, and validate any execution-affecting
change on Linux before Apple. The 32 MiB newer-CUDA workspace remains unmeasured.
