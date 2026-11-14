# Current upstream alignment and TinyStories parser check

The local upstream checkout is clean at the existing pinned commit
`c70274273f489abc73832b2b98584a8483b6b203`. Its `models/vae.py` SHA-256 is
`7d393b56813342278def3b053e83ec4ff4e56a492e4ddf2a2441e1c9f78d4317`.
Source inspection of the supported decoder configuration confirms the same
high-level order: layer normalization, Q/K/V linear projection, optional Q/K
RMS normalization, scaled attention and output projection, residual addition,
then layer normalization, GELU MLP and residual addition. Upstream explicitly
uses decoder `n_layers` as the head count; Rust uses that same field. The
validated configuration requires Q/K normalization and rejects unsupported
encoder/decoder variants rather than silently falling back.

The MLX fast normalization and `addmm` implementations may evaluate FP32
operations in a different order; source correspondence is not bit-exact proof.
The [current Metal qualification](tiny-borrow-apple-gate-v1.md) separately
records 642 trained traces and 2,500 decisions under the existing tolerances.
The Rust untraced reconstruction also reuses the encoded quantized latent,
whereas the independent traced path retains decode construction; snapshot and
output regressions for this intentional optimization are retained in the
[lower-risk evidence](../2026-11-11-phase2-low-risk/README.md).

To check the later TinyStories JSON ownership change, the same standalone Rust
probe was built from clean exports before (`a7c1c22`) and after (`a0bf13c`)
that change in **separate target directories**. On 57 synthetic payloads,
including varied Unicode escapes and malformed schema, hex and merge cases,
both implementations produced byte-identical results: 32 accepted, 15 rejected
as invalid, and 10 rejected as JSON errors. For accepted payloads the probe
compared vocabulary size, six encode/decode cases, special-token handling and
invalid-ID rejection. This compares the two Rust parser versions, not the
Python oracle, and is not exhaustive. The existing four real-profile probes
provide separate pinned-output evidence.

An initial preliminary build used one shared Cargo target and produced identical
probe executable hashes despite different parser sources; that comparison was
rejected as inconclusive. The separately built probes have different hashes and
their compiler dependency files identify the appropriate source trees.
[Corpus, probe, results, hashes and dependency files](raw/tiny-parser-parity-v1.tar.xz)
have SHA-256 `9eb4f89707a81eed9399fecbcae739a3eda9fa60550f3e33e03152c828919fad`.
No production code, pins, fixtures or numerical tolerance changed in this review.
