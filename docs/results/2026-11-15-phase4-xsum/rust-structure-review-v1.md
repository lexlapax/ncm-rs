# XSum Rust structure and memory review

This supplements the [workspace declaration audit](../2026-11-14-phase3-sudoku/rust-structure-audit-v1.md)
with the XSum-specific structures and CUDA operations added afterward. It is a
source review, not proof that every allocation is optimal. Logical tensor
sizes below exclude backend allocator overhead, padding, CUDA caches, weights,
other live activations and transient workspaces.

| Structure or area | Time and space observations |
| --- | --- |
| `XsumModelConfig`, `ContextEncoder`, `Envelope` | Small host-only configuration records, deserialized and compared once during validation. The nine-element vocabulary vector and configuration strings are not per-token inference allocations. |
| `XsumGeneratorInspection` | Retains a 50,304-element FP32 token allowance, 201,216 logical bytes. It is moved into the model after validation rather than retained as a second host copy. |
| `XsumTeacher`, shared `Affine` / `GeneratorBlock` / `TinyHead` | Own checked tensors and immutable model state; blocks and heads become boxed slices after load. Nine 4,096-element FP32 squared-norm tensors total 147,456 logical bytes. The allowance is intentionally also uploaded to the device for the explicit GPU FP64 selector; the host copy supports the separately exposed CPU selector. The request mutex serializes shared-generator work as a correctness safeguard, so these results do not establish concurrent throughput. |
| `LevelGeometry` | Nine cached XSum masks contain 10,281,792 logical bytes and their sine/cosine tensors another 1,230,848 bytes. They are constructed once and reused instead of rebuilding per request; input-dependent padding masks remain request-local. |
| `XsumRun`, `XsumTeacherOutput`, `XsumGenerated` | The run retains final logits until selection; at batch 32 their 32×48×50,304 FP32 elements alone occupy 309,067,776 logical bytes. Output codes are 25,600 logical device bytes and IDs are 6,144 host bytes at batch 32. The traced teacher additionally exposes logits and greedy IDs for diagnostics; these are not a separate production-memory claim. |
| CUDA unique-top-1, masked-softmax and penalty operation records | These small operation records retain native kernel handles, not copies of model weights. At batch 32, the penalty's integer count buffer is 6,438,912 logical device bytes; it also allocates bounded flags, tile partials and output indices. The 48 sequential FP64 penalty steps preserve the pinned decoder order. |

One possible later optimization is that generation reads the concatenated codes
to reject invalid/tied-row sentinels, and the quality consumer reads the codes
again to serialize them. The sampled-embedding contract handles invalid indices
safely, but a change must still reject sentinels before returning, preserve
output semantics, and demonstrate a measured end-to-end improvement. No such
change was made during the source-bound quality run. The `XsumTeacher` and
`XsumGenerated` type comments also predate the device generation path and need
a documentation-only correction after the current source-bound outputs have
been scored.

The measured 1/4/32 latency and whole-process RSS results, device allocator
scopes, background activity and earlier Python OOM are in the
[batch-32 comparison](cuda-b32-gc-comparison-v1.md),
[batch-1/4 comparison](cuda-small-batch-gc-comparison-v1.md) and
[OOM investigation](b32-python-memory-v1.md). Those measurements, not the
logical size calculations here, are the evidence for the observed runtime and
memory direction.
