# Combined candidate Linux gate — 78/78

Candidate `4ab4d3a` includes the approved BERT ASCII normalization and Qwen BPE
allocation changes, plus the Apple-only linear gather and prior shared/CUDA
optimizations. The accepted Rust comparator remains `3307dff`, with the pinned
Python comparators and all limits unchanged. The candidate passed the full
`tools/check.sh --cuda` under umask 077, including 483 report tests and 18
real-CUDA operator cases; all four canonical tokenizer profiles pass. The
previously qualified trained CUDA capture is reused because the codec binary
is byte-identical and subsequent changes affect the tokenizer or Apple backend.
The separate lifecycle capture completed nine processes.

The fresh 45-process matrix initially passed **74/78**. Codec B1 decode,
GPT-2 decode in both modes, and Qwen decode passed point estimates but exceeded
the process-median envelope. One fixed follow-up, declared before collection,
ran nine fresh processes per runtime across all codec, GPT-2 and Qwen scopes and
checked loads; its plan SHA-256 is
`73828bf8f21c95ce6494275802875e9d31a5e24e6028fff9de138ab35e3c3565`.
All 81 processes completed, and the controlled audit passes **78/78**. Against
accepted Rust, follow-up median ratios for those four cells were 0.95750,
1.01549, 1.00617, and 1.00911, respectively. BERT encode passed the initial
matrix at median ratio **0.95670** and p95 ratio 0.95240. No clock policy,
acceptance limit, or adaptive retry was introduced.

[Raw matrix, control, telemetry, quality, lifecycle, probes and checks](raw/final-linux-gate-v1.tar.xz)
have SHA-256 `3f7be3a6828f2b7d680f73701b1c1e6287b017b419b8955eaabf6a092bc29b74`.
The controlled audit SHA-256 is
`f83c53518efd5b8f290dd84742e1644ade04816bec18fd8a3ab7e0ff209ad341`; it
reproduced byte-for-byte with the archived controlled configuration. A 15-second
CPU sample and continuous GPU telemetry are retained; they do not certify a
quiet workstation. Apple final acceptance remains required.
