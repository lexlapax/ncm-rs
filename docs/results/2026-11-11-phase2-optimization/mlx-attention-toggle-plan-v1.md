# Same-process unmasked attention comparison

The existing unmasked-attention candidate removes the scalar zero mask while
preserving bidirectional attention and explicit masks. It previously passed CUDA
identity, Metal checks and 642 trained traces, but separate-process full-codec
measurements were mixed amid activity. This isolated diagnostic alternates the
original zero-mask and unmasked implementations within the same model/process.
The patch adds a diagnostic-only shared toggle; explicit masks continue through
the original path. There is no compilation or production source change.

Verify Linux-qualified production sources and binary identities first. On Metal,
verify original codec and ops source hashes from the [prior plan](mlx-attention-nomask-plan-v1.md),
apply the zero-context patch in a separate checkout, and run three fresh
processes. Each process compares all B1/B2/B4 decode, reconstruct and resident
block scopes, alternating modes for five warmups and 30 measured calls. Check
finite initial outputs within the existing component tolerance and matching IDs,
then require stable IDs in every timed call. Record all timings, first use,
activity and thermal observations. No timing or sample may be discarded.

This is not acceptance: same-process interleaving shares runtime caches and does
not isolate effects on encoder, startup, trained traces or model lifetimes.
The prior independent mixed results remain evidence. Integrate only if the
result, full-codec evidence and unchanged correctness and resource checks
justify it; both unchanged platform gates remain required.
