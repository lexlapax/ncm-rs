# Decoder normalization candidate Linux gate — 78/78

Candidate `621c565` passed the unchanged Linux/CUDA acceptance audit: **78/78**
after one bounded control. The accepted Rust comparator remains `3307dff`,
with pinned Python comparators, workloads, tolerances, and limits unchanged.
The candidate passed the full CUDA check, 18 real-CUDA operator cases and all
642 trained CUDA traces with zero ID disagreements. The fresh three-way matrix
contained 45 processes; the separate lifecycle capture contained nine.

The initial audit passed **73/78**. Five cells met point estimates but missed
the process-median envelope: batch-1 codec decode, GPT-2 decode, Qwen decode
and skip-special decode, and TinyStories skip-special decode. After inspecting
the initial audit and before collecting the control, a fixed plan declared
nine fresh processes per runtime for **every scope and checked load** of codec,
GPT-2, Qwen and TinyStories. This single follow-up completed 108 processes. The
existing paired-process audit passes all 78 cells; initial results and every
follow-up scope are retained. The control plan SHA-256 is
`9d2ca3f73c264d2a0c53061285f117706ae96918190448e991b1983cd6545ff1`.

[Raw matrices, control plan, telemetry, qualification, lifecycle and audits](raw/blocknorm-linux-gate-v1.tar.xz)
have SHA-256 `d487482272e7f52230a2d61f2834d57891cc89474c0f331161a1f22b0c05f842`.
The controlled audit SHA-256 is
`77cf0a9859ffe0bbe15f3f3d50d7777a52250e721a4b18fee29e277f220a410d`.
An export of `621c565` with the archive extracted reproduces the audit
byte-for-byte using the included controlled configuration. The archive was
corrected to retain inputs at the paths specified by that configuration.
An independent source-inventory audit checks all 102 native reports and 153
Python report copies in this matrix and control against clean exports of their
recorded source heads, with zero mismatches. The Python copies represent 51
distinct reports; their bytes match across directories. See the reproducible
[source-audit script](raw/blocknorm-source-audit-v1.py), SHA-256
`43b2e4668739a3cfd976278a2f55594d8749996d981994e4ec8aa75b87498db2`.
The live desktop and sampled activity do not establish exclusive GPU access.
No clock or power policy changed. Fresh Apple validation and its full gate are
still required.
