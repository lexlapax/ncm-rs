# Combined candidate Linux performance gate — 77/78

Candidate `6c95d21` combines the checked finite scan, BERT vocabulary
reservation and packed decoder Q/K/V layout. The saved accepted Rust comparator
is `3307dff`; the pinned Python comparator and all performance bands are
unchanged. `tools/check.sh --cuda` passed, including 482 reporting tests, real
CUDA tests and 18 operator cases. Its log SHA-256 is
`5c80c6749e881ae011cb53460dab0cd98758ea09ccc85c56c6094a65c53e8089`.
The prior intermittent CPU lifecycle failure remains historical; this passing
run does not establish its cause. The packed candidate's independently retained
trained CUDA qualification passes 642 traces / 2,500 decisions and has the
same native trace hash as the earlier candidate.

The fresh three-way matrix completed **45 processes** and the separate lifecycle
capture completed nine. The initial audit passes **77/78**: Qwen decode versus
accepted Rust passes point estimates but fails the three-process envelope.
The previously open B1 block and BERT checked-load cells pass: candidate/previous
median ratios 0.89743 and 0.63019, respectively. All Python and comparable host
memory comparisons pass in the initial audit.

One nine-process-per-runtime Qwen follow-up, covering all Qwen scopes and checked
load, was declared before launch; no retry or threshold change was allowed. Its
plan SHA-256 is
`b06c5cf69e86baee63c3a93a695662692502ea3295be19e9e998c80047c31627`.
All **27 processes** completed. Qwen decode now passes, but Qwen checked load
versus accepted Rust fails: the follow-up median ratio is **1.05535** against
the 1.05 limit, with 95% interval **[1.03894, 1.06928]**; its p95 ratio is
1.04086 with interval [1.02841, 1.05958]. All nine candidate process medians
were 2.6–8.2% slower than their paired previous processes. The unchanged
controlled audit therefore also passes **77/78**, with that single cell open.
The prior controlled 76/78 result for `a6290db` remains historical; this is not a
passing Linux or Apple gate.

Before the matrix, the GPU reported 0% utilization; a 15-second read-only
sample measured CPU 2 median 1.98% / maximum 8.0% non-idle and SMT sibling
CPU 3 median 0% / maximum 2.97%. This neither certifies a quiet window nor
establishes conditions during the entire collection. No system policy changed.
[Raw matrix and control reports, telemetry, fixed plan, lifecycle, quality,
CUDA check and audits](raw/combined-linux-gate-v1.tar.xz) SHA-256:
`cb98da607fdd11155b0801ddb376e43516afb9716484f2df38cc54a9fbca0ecb`.
The controlled audit SHA-256 is
`23e65a60a35e53a120710dd03282692255d1ddd5f2b9a82fe86fa7db9bf611e8`.
Extract the archive into an empty directory and run
`python3 /PATH/TO/REPO/tools/bench/audit_phase2_performance.py
reports/phase2-combined-linux-gate-v1/controlled-config.json NEW_OUTPUT`;
exit 1 and the single failure are expected. The output was reproduced
byte-for-byte in `/tmp/ncm-combined-gate-repro-v1`.

Next, investigate Qwen checked-load work against accepted Rust; do not discard
the control, rerun it adaptively, or proceed to Apple before the Linux gate.
