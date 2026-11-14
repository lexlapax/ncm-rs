# Approved byte-level candidate Apple gate — 76/78

Candidate `3a3fa86` has the runtime code of Linux-qualified `1f7d2a8`. Its
exact tokenizer source passed the full MLX check in the isolated review
worktree and all four fresh tokenizer probes pass. The Metal codec benchmark,
trace and lifecycle binaries are byte-identical to those previously qualified
with 642 trained traces / 2,500 decisions. The 45-process three-way matrix and
separate nine-process lifecycle use the unchanged Rust `3307dff` and pinned
Python comparators, fixtures and limits.

The initial audit passes **76/78**. GPT-2 encode now passes, with median ratio
**0.86573** and p95 ratio **0.92982** versus accepted Rust. The two initial
uncertainties were batch-4 resident block versus Python and Qwen skip-special
decode versus Rust. A single predeclared, nine-process-per-runtime control
covered every codec and Qwen execution scope and checked load. The controlled
audit remains **76/78**: the block now passes, while Qwen decode and Qwen
skip-special decode have paired-process median interval upper bounds
**1.05109** and **1.05544**, respectively, against the unchanged **1.05**
limit. Their controlled median ratios are **1.00063** and **0.98507**. Qwen
decode passed the initial envelope but is evaluated by the full control.

At the user's request, activity was observed before and during collection;
the user stopped desktop use and closed Chrome. The initial matrix recorded
106 two-second samples: WindowServer appeared in all, median reported CPU
9.35% and peak 30.9%. The control recorded 275 samples: WindowServer appeared
in all, median 9.1% and peak 38.0%. Neither Chrome nor `rmlx` appeared among
the sampled busiest processes. Thermal reporting recorded no warning before
or after either run. These are sampled process observations, not proof of GPU
exclusivity or a measurement of GPU clock behavior. No clock or power policy
changed.

[Raw matrix, control, plans, activity, quality, lifecycle and audits](raw/bytelevel-final-apple-gate-v1.tar.xz)
have SHA-256 `2ba445f0eb3b37b6b5920078a5ed044546b824a78edd21b335e04a263df27904`.
The controlled audit SHA-256
`e019e4b3dfdbd35d561b5c00b929864bb8cef3e80200c135744326e299f57857`
reproduces byte-for-byte, including the two failures, from a clean `3a3fa86`
export and the archive. All 66 native report inventories and 99 Python report
copies match clean baseline or candidate exports. Python reports are shared
copies, not independent samples. Phase 2 remains open.
