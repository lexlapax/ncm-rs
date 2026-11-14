# TinyStories loader borrowing candidate

The TinyStories byte-BPE loader now borrows unescaped hex strings from the
already authenticated JSON payload while building its owned vocabulary and
merge table. Escaped JSON strings still deserialize into owned strings. No
borrow escapes the constructor; byte-BPE encoding, decoding, rank order,
validation rules, asset schema and pinned upstream behavior are unchanged.
A regression compares escaped and unescaped forms through encode and decode.

The isolated candidate passed `tools/check.sh --cuda` under umask 077 with
`RUSTC_WRAPPER=`, including 18 real-CUDA operator cases, the tokenizer suite,
and four real canonical tokenizer probes. In a separate instrumented Linux
allocation diagnostic, median checked-load Rust allocation requests fell from
55,033 to 30,973 per load; requested allocated bytes fell from 6,114,737 to
5,930,549. These are allocator call counts, not live-memory measurements.

Three alternating fresh pairs using the pinned TinyStories workload showed
median-of-process checked-load time **3.35390 to 2.88641 ms**. Observed Linux
load HWM values were 11,325,440 / 11,120,640 / 11,108,352 bytes current and
10,416,128 / 10,366,976 / 10,268,672 bytes candidate. Encode median ratio was
1.01429; decode and skip-special median ratios were 1.04035 and 1.06957.
Output hashes match. These small diagnostic samples are not acceptance results,
and the apparent changes in execution scopes require the complete fixed gates.
No Apple memory reduction is claimed yet.

[Raw comparison, allocation counts, source patch, probes, full check and CUDA report](raw/tiny-borrow-diagnostic-v1.tar.xz)
have SHA-256 `88905228b40d00869bdf60eb5395592bab0ff5e5662c0093db88a2f2eef60536`.
All six timing and both allocation report source inventories match their
respective source trees. The earlier Apple 76/78 result remains retained.
Run a fresh Linux gate first, then the full Apple checks and gate if Linux
passes, with no changes to acceptance limits.
