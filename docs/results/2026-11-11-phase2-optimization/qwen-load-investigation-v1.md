# Qwen checked-load investigation

The combined candidate's fixed Qwen control failed checked-load median against
accepted Rust (ratio 1.05535); all nine paired process medians favored the
accepted binary. This investigation does not retry or replace that control.

One allocation capture per binary showed identical median Rust GlobalAlloc
work per checked load: 758,101 allocations, 60,376 reallocations, and 55,700,878
requested allocated bytes. These counters exclude native allocations, allocator
internals, RSS and physical memory; they do not prove the two binaries do equal
CPU work. The initial baseline capture could not publish because the historical
checkout lacked the added diagnostic source. A second capture used an isolated
archive of `3307dff` with the existing diagnostic sources copied in; the saved
historical profiler binary was unchanged.

A separate instrumented candidate run added timers around canonical BPE
deserialization and the pinned builder. Across 30 measured loads in one
process, medians were 26.352 ms for parse and 32.615 ms for construction, out
of 67.975 ms instrumented total checked load. Instrumentation changes timing
and cannot be used as an acceptance comparison. The instrumentation patch was
reverted. The payload has 786,875 literal Unicode escape introducers; a standalone
in-place probe converted 786,866 safe non-surrogate, non-ASCII BMP escapes in
16.789 ms median, reducing the buffer from 14,570,119 to 11,422,655 bytes.

An isolated candidate applied that safe in-place preprocessing only after the
original Qwen payload passed its pinned hash. JSON-equivalence tests covered all
non-surrogate BMP scalars, control escapes, escaped backslashes, surrogate pairs
and invalid escape cases; the real four-profile tokenizer probe passed. In three
matched processes per binary, median checked load increased **65.693 to 77.133
ms** (ratio 1.17415); encode and both decode output hashes matched. Host HWM
was roughly 76 MB in both sets. The candidate was rejected and reverted without
repeating it. No payload, schema, pin, dependency or runtime API changed.

The [raw allocation captures, instrumentation patch/profile, standalone probes,
candidate patch, matched reports and parity probe](raw/qwen-load-investigation-v1.tar.xz)
have SHA-256 `37309157b2396c774125d08d43c0d224ba71b76c90e4a6f182faa52c3e0e8a0b`.
They contain no model assets or binaries. The production release `asset_bench`
was rebuilt after the revert and matches the prior candidate executable SHA-256
`0bd35209d79a3c0f9fe2b7d1a026cea99e319b385feba7af5bc496bf89f858ba`.
No Qwen acceptance cell closes; Linux remains 77/78 for the combined candidate.
Any change to the pinned dependency or its public construction API requires
separate review under the approved plan.
