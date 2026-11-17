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
