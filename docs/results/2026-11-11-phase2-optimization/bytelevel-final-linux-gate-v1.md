# Approved byte-level candidate Linux gate — 78/78

Candidate `1f7d2a8` passes the unchanged Linux/CUDA acceptance audit **78/78**
on its initial 45-process three-way matrix, without a follow-up. The accepted
Rust comparator remains `3307dff`; the pinned Python comparator, artifacts,
workloads and limits are unchanged. A separate nine-process lifecycle capture
and read-only CPU/GPU telemetry were collected. GPT-2 encode's median ratio
versus accepted Rust is **0.93736** and p95 ratio is **0.92678**.

All four pinned tokenizer asset probes pass. The exact source passed the full
CUDA check in the isolated review worktree; main-workspace tokenizer tests and
formatting also pass. The codec benchmark, trace and lifecycle binaries are
byte-identical to the previously qualified `621c565` binaries, so the 642
trained CUDA trace / 2,500 decision qualification is retained. This reuse is
limited to those unchanged codec executables; the tokenizer matrix and probes
are fresh.

[Raw matrix, plan, telemetry, probes, lifecycle, qualification and audit](raw/bytelevel-final-linux-gate-v1.tar.xz)
have SHA-256 `90ed108c28f63d5196e1df3f55abf6dd63a3943d9d765ef6820249f2af38bc95`.
The audit SHA-256 is
`26fd857f97d1183805e86f43dcb7a831b4daff74f2a8e39c914f0393e2eac105`;
an extracted archive and clean `1f7d2a8` source export reproduce it byte for
byte. All 30 native report inventories and 45 Python report copies match clean
candidate or baseline exports. Python copies are shared, not independent
samples. The live desktop and activity sample do not establish exclusive GPU
access. No GPU clock or power policy changed. Final Apple validation remains.
