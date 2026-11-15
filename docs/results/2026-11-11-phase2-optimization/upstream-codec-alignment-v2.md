# Retained codec alignment with pinned Python

This source review addresses the request to stay close to upstream algorithms.
The local upstream checkout was clean at pinned commit
`c70274273f489abc73832b2b98584a8483b6b203`; the retained production sources
match gated candidate `5200924` under the [binding check](phase2-closure-bindings-v1.md).
It is an inspection of the supported inference codec, not a claim of parity for
all upstream configurations, training, stochastic encoding or generation.

| Area | Pinned Python behavior | Retained Rust behavior and distinction |
| --- | --- | --- |
| Encoder input | Token embedding plus learned positional embedding in eval mode. | Uses the full validated immutable position table directly instead of gathering every row. The supported input spans that table; this is not a general variable-length positional API. |
| Quantizer distance | `_distances` computes squared-input norm plus squared-codebook norm minus twice the dot product, then deterministic `argmin` in the non-stochastic path. | Keeps the row-constant input norm and operation order at the shared-model level; caches the parameter-only codebook norm per model. Backend kernels can round differently. |
| Level geometry | `_down_for_quant` uses the learned level-zero cascade or area interpolation; `_up_to_full` uses the learned level-zero cascade or linear interpolation. The terminal level uses full-length values. | Same supported level/cascade choices. CUDA and MLX implement the operations with native geometry and layouts; numerical equivalence is checked against the unchanged tolerance, not asserted bitwise for arbitrary inputs. |
| Residual refinement | `Phi` computes `(1-r)*input + r*conv(input)`; encoding adds each contribution to the cumulative value and subtracts it from the residual. | Uses the same expression and update order for the validated inference configuration. The first contribution initializes the accumulator without a separate zero tensor; signed-zero and backend rounding are not assumed identical in all hypothetical inputs. |
| Decoder block | `models/vae.py` uses sequential pre-LayerNorm residual attention followed by a GELU MLP, with Q/K normalization in the qualified configuration. Its unmasked attention is bidirectional. | Keeps that block order and unmasked codec semantics. Q/K/V are packed for layout efficiency; native normalization, attention and biased projections may use fused operations and different FP32 evaluation order. `models/common.py`'s causal default is not the codec reference. |
| Reconstruction and snapshots | `VQVAE.reconstruct` unpacks three values from a quantizer `forward` that returns five; `idx_to_zhats` appends a mutable accumulator subsequently updated in place. | The [predeclared reference contract](../../PHASE2.md) avoids those upstream helper defects: decode is checked against `idx_to_z_q`, and shadow sums supply independently retained snapshots. Untraced Rust reconstruction reuses the encoded cumulative latent, while the traced route constructs decode independently; existing snapshot and output regressions cover this distinction. |
| Text tokenizers | Pinned upstream assets and HF behavior determine IDs, byte handling, special tokens and cleanup. | The user-approved vendored patches target specific allocation/identity cases without changing the tokenizers version. The [scope audit](dependency-scope-audit-v1.md) and four-profile probes retain the evidence and limitations. |

Relevant upstream files: [`models/quant.py`](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/quant.py),
[`models/vae.py`](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/vae.py)
and [`models/vqvae.py`](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/vqvae.py).
Their local SHA-256 values are respectively
`5fd16053a826a5a0af580715a79bb03dc3eea49669514969c1086c445e9c9e2b`,
`7d393b56813342278def3b053e83ec4ff4e56a492e4ddf2a2441e1c9f78d4317` and
`9873af280da08ca39360388f652f993f17ace761a751ffa812caa706f66f592b`.
The pinned remote `quant.py` was also fetched read-only and has the same hash
as the local Git object and checkout file.

The archived platform qualifications separately cover **642 traces and 2,500
selection decisions**, with no disagreements under the pinned numerical policy;
they do not prove exhaustive equivalence. The current operation-level fusion
experiment remains [deferred](mlx-linear-fused-codec-result-v1.md) and is not
part of production. No upstream pin, artifact, schema, oracle or tolerance was
changed by this review. The separate live-memory question remains open.
