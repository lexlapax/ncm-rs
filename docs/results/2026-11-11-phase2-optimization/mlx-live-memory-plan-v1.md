# Matched MLX retained-root memory comparison

The [retained snapshot review](mlx-active-snapshot-review-v1.md) found candidate
post-block active-allocation snapshots above the 1.05 comparable live-memory
band if the roots and cleanup state are equivalent. The 78-cell latency/RSS
audit does not decide that separate requirement. This predeclares a focused
comparison after the current gate completes and only if its controlled audit
passes; do not run it concurrently with the gate.

Use three fresh processes for each of accepted Rust `3307dff`, the retained
candidate, and qualified adapted Python, rotating process order. The Rust and
Python probes use the same qualified artifact, fixture-derived tokens/codes,
B1/B2/B4 order and encode/decode/reconstruct/resident-block scope order. Each
scope performs one first call and 35 subsequent calls, checks stable output IDs,
then measures MLX active/cache/peak bytes with the same model and evaluated
latent retained and the returned output dropped. Record four snapshots: immediate,
after a synchronous same-device barrier, after 10 ms and after 100 ms. No cache
is cleared. Record the evaluated latent's float-byte hash, input/code/output ID
hashes, artifact manifest, exact commands, source/build/binary identities,
activity and thermal observations. All rows and observations remain;
this is a memory diagnostic, not a latency or 78-cell acceptance run.

The Rust barrier is a synchronous operation on the backend stream; Python also
calls `mx.synchronize`. Waiting is a predeclared diagnostic to observe cleanup,
not permission to exclude the immediate data. After the batch scopes, drop the
latent; after the model, barrier and scalar are dropped, sample active memory
again. Cross-runtime live graphs may still differ despite matching retained roots;
MLX active bytes are not host RSS or separate VRAM. Require manifest, input and
output identities, qualified Python baseline and expected role/source bindings.
If float hashes differ, outputs mismatch, reports are missing, memory remains
unsettled, or retained graphs cannot be shown comparable, report the limitation
rather than declare the 1.05 live-memory requirement passed. Compare all batch
and scope observations to the unchanged band; do not select a favorable delay.
The existing [candidate-only lifetime observation](mlx-memory-addmm-v1.md) and
prior failure evidence remain intact. No production code, clock, power policy,
pins, assets or tolerances change.
