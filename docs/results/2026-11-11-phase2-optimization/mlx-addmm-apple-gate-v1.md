# MLX projection candidate Apple gate

Candidate `a323b16` has a **74/78** initial Apple audit and **76/78** after the
predeclared all-workload control. The pinned trained Metal qualification passed
642 traces and 2,500 decisions with zero disagreements, and measured codec and
trace binaries match the qualified experimental builds. Linux's 78/78 evidence
remains applicable to the five byte-identical Linux executables. Nine Apple
lifecycle processes agree on output IDs. Phase 2 remains incomplete.

Both remaining cells are TinyStories host memory versus the previous Rust
baseline: VmRSS and VmHWM peak ratios are **1.12434**, above the 1.05 limit.
The candidate's largest final snapshot is 17,334,272 bytes versus the previous
runtime's 15,417,344 bytes. In each group that maximum comes from its third
process; those processes already had higher snapshots immediately after the
load trials (16,891,904 versus 15,089,664 bytes for HWM). The observation does
not establish the cause. Batch-4 block and Qwen encode cells pass this control.

The initial matrix had 107 activity samples and the control 320. Chrome did not
appear in top-process entries. WindowServer median/peak CPU was 9.5%/38.0%
initial and 10.4%/33.7% control. No thermal warning was reported before or after
either run. CPU snapshots do not establish GPU exclusivity or explain memory
variation; no clock or power policy was changed. Prior failing gate results are
retained, and the unchanged candidate must not be retried as a substitute for a
substantive fix.

[Raw reports, plan, lifecycle, activity, audit and source verification](raw/mlx-addmm-apple-gate-v1.tar.xz)
have SHA-256 `c999f4c65bb4d19499ee9bf122d9e33bb05797279e1387dd133d91e8d280e0a2`.
The controlled audit SHA-256
`718170766dbea885ad75f350583cf4913eeadf274b3815e15d899a2fb98b469d`
reproduces byte-for-byte from a clean candidate export. All 120 native report
inventories and 180 shared Python report copies match clean exports; the core
trait source omitted from the standard codec inventory is separately bound.
The prior four-profile tokenizer probe is retained for the unchanged tokenizer;
the corresponding Linux probe and benchmark binaries are byte-identical.

A [separate finite MLX memory observation](mlx-memory-addmm-v1.md) covers the
current runtime and does not change these host-memory acceptance failures.
Investigate TinyStories loader allocation while preserving the byte-BPE
algorithm, asset validation and existing limits.
