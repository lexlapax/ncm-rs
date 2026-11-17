# WMT14 upstream KV-cache diagnostic

The [synchronized stage profile](cuda-stage-profile-v1.md) shows that
repeated transformer work dominates the measured Rust intervals. The pinned
upstream implementation exposes `use_kv_cache=True`, but its cache path omits
the prefix padding bias both when populating cached prefix keys/values and in
the later level mask. The released WMT14 prefixes contain padding, so this
path cannot be assumed semantically equivalent to the qualified no-cache
reference. No runtime switch or reference change is implied.

A separate, pinned-source Python diagnostic will run the unmodified upstream
cache option on the first 32 prefixes, with the same FP32, math-SDPA and timing
scope as the existing benchmark. It will compare every code and raw ID to the
retained no-cache Python capture and preserve failures, timing and source/input
hashes. Even a match on this small sample does not qualify the path or prove
padding correctness. If it differs, those differences are evidence against
using the upstream cache path unchanged. Linux CUDA runs precede any Metal work.

The unmodified pinned upstream cache option completed the batch-32 probe. All
32 prefixes contain padding. Relative to the retained no-cache Python output,
7,505 of 10,016 codes and 1,005 of 2,048 raw IDs differed across all 32 rows.
The single-process cache median was 550.24 ms versus 1267.83 ms for one
previous no-cache process; these are exploratory timings and not a qualified
speedup. The observed mismatch is consistent with the missing padding masks,
but this test does not prove that they are the only cause. The output SHA-256
is `0c3b10b67f8d5c68ffbb0bf5bf4533c8b5f5340c18c604203f17574403aa269d`.

A separate diagnostic adaptation will install temporary forward hooks that add
the omitted prefix-padding bias only while upstream caching is enabled. It
supports CFG=0 only, leaves the pinned checkout untouched, and explicitly
records this correction rather than presenting it as upstream behavior. The
same 32-row no-cache capture remains the comparison. This is an exploratory
reference fix: exact outputs, speed and broad quality remain to be measured,
and no implementation or tolerance change is assumed.

The explicitly padding-corrected diagnostic matched all 10,016 codes and
2,048 raw IDs in the 32-row no-cache sample. Its single-process median was
552.31 ms versus 1267.83 ms for the earlier no-cache process; this is not a
repeated or exclusive-device performance comparison. The output SHA-256 is
`6711d08ac95de2ef71e034c20d88875472031a470908544643344cfb76c3a6a6`.
The hooks were applied only during cached attention and removed afterward;
the pinned upstream checkout and frozen reference remained unchanged.

An explicitly selected Rust cache candidate is planned. It will retain
prefix K/V from each block after processing the padded prefix with the
qualified mask, apply that padding bias to each target query, use the same
rotary positions, and never reuse a previous target's K/V. The default
no-cache path remains available and unchanged. First validate exact outputs
and numerical bands on real CUDA at batch boundaries, including padding and
CFG; full corpus, frozen quality and repeated performance/memory evidence are
required before qualification. This path is a new implementation, not a
silent adoption of the incorrect upstream cache mode, and the 32-row result
is on previously inspected data.

## Explicit Rust cache candidate preparation

The standard `Wmt14Teacher` constructor remains no-cache; an explicitly named
experimental constructor reuses only per-block prefix K/V, retains the
qualified prefix padding mask, applies it to target queries, and does not reuse
a previous target's K/V. The 35-row CUDA smoke (batch 4 plus a 3-row tail)
matched all codes and raw IDs in both base and CFG modes against retained
Python outputs. The first comparison attempt used system Python without Torch
and failed before comparison; the pinned reference environment rerun passed.
The base and CFG capture SHA-256 values are
`ea9c4b860f5fa8100510388df0663a9e0b9af0f64e502bdbc1bbb432a91d3f5d` and
`783ad81500a6b8a86247eed9c8544b02f7c52fbd9307728c916ed4da61272d27`.

An explicit ignored local-artifact CUDA test was run: cached and uncached
outputs matched for batches 1/4, base/CFG, on ordinary padded rows and
synthetic unpadded, null and mixed interior-padding inputs. Its log SHA-256 is
`78d35521d3db6df1a33d2fc2391afdaf76fe717d9d5a3811ff568e308dfa7701`.
A single-process batch-32 smoke matched retained Rust and Python codes/IDs;
its cached Rust median was 535.73 ms versus earlier uncached single-process
Rust/Python medians 1280.53/1267.83 ms. This is exploratory, not a repeated
performance gate. Full CUDA checks, full-set replay and quality, repeated
performance/memory and Metal validation remain pending; no reference or
tolerance has changed.
The full local CUDA check passed, including 18 operator cases; log SHA-256
`9c58cb4e32091910838446a9a5d4b73fe048d160a6593c4e6f32f6c2390028eb`.
The feature-enabled release regression and source-bound full captures remain
pending. The candidate's separate capture and benchmark supervisors are
prepared; they reuse the frozen references and enforce source/input postflight.

The feature-enabled release suite passed (log SHA-256
`70bb452a94c4901862998a20a53d64dcc23230b8f9e48e88c07acf9911c1dd7e`).
The final explicit CUDA regression rerun passed both tests: exact cached versus
uncached codes/IDs for padded, unpadded, null and interior-padding inputs, and
all 23 forced-history tensor traces within the existing comparison bands,
with matching shapes and all three traced index arrays. It used four previously
inspected rows and does not replace full-corpus qualification. Its log SHA-256
is `72b455bfe6869ebd0f173708eb3fbc85f7b091c7bc8f0bc2efba7289297adab2`.
