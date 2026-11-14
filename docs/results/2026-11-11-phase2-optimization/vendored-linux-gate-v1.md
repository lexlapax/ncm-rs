# Vendored BPE candidate Linux gate — 78/78

Candidate `582e8fe` includes the approved pinned BPE merge-allocation change,
the BERT vocabulary reservation, finite scan and packed Q/K/V. The unchanged
accepted Rust comparator is `3307dff`; the pinned Python comparators and
acceptance bands remain unchanged. `tools/check.sh --cuda` passed with umask
077, including 482 reporting tests and 18 real-CUDA operator cases. An earlier
run under umask 022 failed three existing clock-evidence fixture tests before
the CUDA section; both logs are preserved. The unchanged packed-Q/K/V trained
CUDA qualification is retained with its 642 traces and 2,500 decisions.

The fresh matrix completed 45 interleaved processes and a separate nine-process
lifecycle capture. Its initial audit passed **75/78**: Qwen checked load versus
accepted Rust had median ratio **0.95851**, p95 ratio 0.94408, and passed. The
three pending cells were codec B1 decode and Qwen decode in both skip modes,
whose point estimates passed but process-median envelopes exceeded 1.05.

One follow-up was declared before launch: nine fresh processes per runtime for
each of codec and Qwen, including every execution scope and both checked loads,
with CPU 2 and observer CPU 0. The plan SHA-256 is
`73823b0df06cf9cb1579db1606e760d70dea70ac51c2641f6bd04eed3897c46d`.
All 54 processes completed, and the controlled audit passes **78/78**. Against
accepted Rust, follow-up median ratios were 0.95838 for B1 decode, 0.98003 for
Qwen decode, 0.97193 for Qwen decode with special tokens skipped, and 0.95573
for Qwen checked load. Their interval checks and the other cells pass the
unchanged limits. No adaptive retries, clock changes, or threshold changes were
made. This Linux result does not establish Apple/MLX parity or performance.

[Raw matrix, controls, logs, quality, lifecycle, plans, telemetry and audits](raw/vendored-linux-gate-v1.tar.xz)
have SHA-256 `d403f314a70f7fbba1a9a0652ea5ee3aa7a8798ef66652e9070bce8fbe67750c`.
The controlled audit SHA-256 is
`94f59cf5247788ce0abcca45fcef0e13f419d5cb0fca9723581866071d0e45f7`;
it was reproduced byte-for-byte with `tools/bench/audit_phase2_performance.py`
and the archived `reports/phase2-vendored-linux-gate-v1/controlled-config.json`.
A read-only 15-second CPU sample and GPU telemetry are included; these do not
certify a quiet window. The GPU showed 0% utilization before the matrix.

Apple is next. An initial noninteractive connection reached `macstudio` and
the server accepted the configured public key, but the 1Password SSH agent
refused to sign; remote availability and checkout state are still unverified.
