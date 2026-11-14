# MLX affine candidate Apple gate

Candidate `2850331` has a **77/78** initial Apple audit and **75/78** after the
predeclared unconditional all-workload control. The pinned trained Metal
qualification passed 642 traces and 2,500 decisions with zero disagreements;
the measured codec and trace binaries are byte-identical to the qualified
experimental builds. Linux's 78/78 evidence remains applicable to the five
byte-identical Linux executables. Nine Apple lifecycle runs agree on output IDs.
Phase 2 is incomplete.

| Pending cell / comparison | Controlled evidence |
| --- | --- |
| Batch-4 resident block / previous Rust | median ratio 0.96930; p95 interval upper 1.15616 > 1.10 |
| Batch-4 resident block / Python | median ratio 0.95812; p95 interval upper 1.15006 > 1.10 |
| Qwen encode / previous Rust | median ratio 1.02137; median interval upper 1.05366 > 1.05 |

The initial audit's only pending cell was GPT-2 encode versus previous Rust;
it passes in the controlled audit. The earlier candidate's 72/78 result is
retained. No unchanged-candidate retry is justified by these outcomes.

The initial matrix had 106 activity samples and the control had 318. Chrome did
not appear in their top-process entries. WindowServer median/peak CPU was
11.35%/35.6% initial and 10.0%/35.9% control. No thermal warning was reported
before or after either collection. These CPU snapshots do not establish GPU
exclusivity or identify the cause of tail variation; no clock or power policy
was changed.

[Raw reports, plan, lifecycle, activity, audit and source verification](raw/mlx-affine-apple-gate-v1.tar.xz)
have SHA-256 `73ea5566da0e9f4026cc7358f8b291d37fcf0295945bd5c0b4b0ef3102677bb1`.
The controlled audit SHA-256
`4cbb1ddf9e9c80254bf15ccc14125580ebd2300010408708dd2ef4dbfe859b4b`
reproduces byte-for-byte from a clean candidate export. All 120 native report
inventories and 180 shared Python copies match clean source exports. The core
trait file omitted from the standard codec inventory is separately bound by
its compiler dependency file and source hash to the measured binary. The
retained tokenizer probe is from the prior candidate; the tokenizer probe and
benchmark binaries are byte-identical on Linux and the tokenizer source is
unchanged on this MLX-only candidate.

Next investigate the two block p95 cells and Qwen encode, preserving upstream
semantics and Linux-first validation. The bounded long-lived MLX cache evidence
also remains open; allocator snapshots alone do not establish it.
