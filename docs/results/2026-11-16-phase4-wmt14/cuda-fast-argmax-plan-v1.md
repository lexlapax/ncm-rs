# WMT14 CUDA final-selection candidate

The uncached WMT14 reference backend's generic last-axis first-maximum
selection constructs full-logit equality and index intermediates. On the
batch-32 generated logits (`[32, 64, 151680]`), an isolated, synchronized
20-sample diagnostic measured median 94.050 ms for this selection alone. The
existing level-stage diagnostic measured a 125.370 ms decoder-plus-selection
checkpoint. A separately instrumented pinned Python decoder measured 24.957 ms;
its decoder hook excludes the subsequent argmax, so these are not directly
comparable complete decoder timings. Both profilers reproduced the retained
batch-32 codes and IDs; they run in separate processes and include different
synchronization, and desktop GPU activity was not excluded. They are
bottleneck diagnostics, not headline performance evidence.

A bounded opt-in CUDA operator reads each 151,680-value row once and returns
the first maximum without allocating full-logit temporary arrays. Its explicit
backend constructor is separate from the existing reference backend. The
one-pass kernel ignores NaNs in the same way as Candle's FP32 `fmax` reduction;
all-NaN rows retain the generic `u32::MAX` sentinel. Real-CUDA tests compare
the candidate and generic operator on 64 full-width rows including ties across
threads, infinities, all-negative-infinity, all-NaN, mixed NaN/negative
infinity, signed zero, and deterministic unique maxima. Unsupported shapes
are rejected by the explicit operator. The opt-in backend selects this operator
only for largest-index selection at `[1..=32, 64, 151680]`; other shapes and
default constructors retain the existing path.

On retained generated logits, the explicit candidate and generic selection
returned identical IDs. Separate one-process synchronized selection screens
measured candidate median 2.927 ms versus the generic 94.050 ms. A preliminary
30-trial full-generation screen measured candidate batch-32 median/p95
1.183275/1.187579 s and matched retained codes and IDs; its process and
conditions differ from earlier Python measurements, so it is not a paired
performance comparison or gate pass. The first CUDA integration test used a
tensor from a separate device context and correctly failed; the corrected
context test, targeted CUDA tests and feature-specific Clippy passed. The full `tools/check.sh --cuda` rerun passed including 18 device cases
(`reports/phase4-wmt14-fast-argmax-cuda-check-v2.log`). Its first run failed
the process-lifecycle fixture `test_real_writers_and_owned_process_samples_compose`;
the isolated retry and unchanged full rerun passed, and the failed log is
retained. Targeted real-CUDA integration and feature-specific Clippy/build
pass. Source-bound full-corpus comparisons, frozen quality scoring, and
repeated same-condition performance remain pending.

Evidence retained under ignored `reports/`:

- Generic selection result SHA-256 `d244eff38c00db6a86360d1d225c3adcb4f9a4d12cf2530b53edca614e04bf04`.
- Candidate selection result SHA-256 `127bbb9277c0b8545eb41503ec6e9057b1266a4f0eace5d9a815f24326f02aee`.
- Candidate full-generation screen SHA-256 `d8faf686394a5f77bd2bdc898e3de89eab1a51857a5407faa161113d66837cfc`.
- Rust stage profile base/CFG result SHA-256 `d49c9dfee4b84a1876479ec0757bc61b98a383256911d6731904244dff8bf352` / `94a7533185c6ecc1c6dc7b56d983176ed7e124c2783b53606e3852fa3adae177`.
- Pinned Python all-level and decoder profile SHA-256 `5375815c347292694c9dc4247d1d98f6315feadda7131952daec3ec21c5c962b`.

The CUDA activity and timing limitations above remain; no cross-platform or
full Phase 4 acceptance is claimed.
