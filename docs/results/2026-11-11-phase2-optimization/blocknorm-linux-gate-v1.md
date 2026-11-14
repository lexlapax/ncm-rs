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
have SHA-256 `3f7725b7dc651d965d247018b83da80d15f0aaf27ebdf16e047883747e6b0dcf`.
The controlled audit SHA-256 is
`77cf0a9859ffe0bbe15f3f3d50d7777a52250e721a4b18fee29e277f220a410d`.
The live desktop and sampled activity do not establish exclusive GPU access.
No clock or power policy changed. Fresh Apple validation and its full gate are
still required.
