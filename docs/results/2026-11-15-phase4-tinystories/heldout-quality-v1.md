# TinyStories CUDA held-out stochastic quality

The [criteria](development-quality-v1.md) were committed as `d60ef4b` before any official-seed samples were generated. Both pinned Python and experimental Rust generated 256 samples per seed at the upstream generation batch 128 on an NVIDIA GeForce RTX 3060 Ti. Their sampling distributions target the same temperature-1 categorical probabilities but use different random-number streams and algorithms; identical seeds are not expected to yield identical samples. All raw IDs, decoded text and re-encoded tokens passed validation against the pinned upstream BPE tokenizer.

| Seed | Python genPPL | Rust genPPL | Python entropy | Rust entropy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16.5725 | 16.2356 | 3.966081 | 3.972471 |
| 1 | 15.6596 | 16.1946 | 3.964875 | 3.971008 |
| 2 | 16.4025 | 15.8858 | 3.976839 | 3.959304 |
| 3 | 16.3518 | 16.2341 | 3.973933 | 3.970265 |
| 42 | 16.1760 | 16.1364 | 3.959324 | 3.967674 |
| Mean | 16.232494 | 16.137318 | 3.968210 | 3.968144 |

The Rust mean PPL is 0.095176 lower, within the frozen upper non-inferiority margin of 1.15; absolute entropy difference is 0.000066, within the frozen 0.019 margin. **This stochastic-quality component passes**, subject to the scorer qualification described below. It does not establish exact sampled outputs, full numerical parity, speed/memory gates, Apple/MLX behavior or completion of Phase 4.

The unmodified upstream GPT-2-large scoring function OOMed at batch 32 on this 8 GiB GPU. The scored run used the documented [memory-bounded evaluator](../../PHASE4-TEXT.md): it keeps batch-32 input and transformer execution, suppresses unused KV-cache output and chunks the output head/log-softmax. Its earlier development qualification matched all upstream token counts and had maximum per-token mean-NLL difference `4.806e-7` against the unmodified function at batch 8. The independent qualification does not directly compare unmodified batch-32 outputs because that run did not fit. The GPT-2-large model/tokenizer and Transformers version are pinned in the Phase 4 contract. Source sample JSON and scoring output remain outside Git in `reports/phase4-tinystories-heldout-v1/`; `quality-v1.json` SHA-256 is `a5ba3ba02d5fa49ee758692b43957c4e5d73ecfc0e83c9986d23e7b694a7e2fb`. The scorer's process peak RSS was 4,299,356 KiB; that is evaluation memory, not generator memory.

## Strict cuDNN candidate retest

After the result above, a development-trace investigation found material FP32 convolution differences and led to the explicit strict-cuDNN candidate described in [the numerical investigation](cuda-numerical-investigation-v1.md). The change was motivated by numerical behavior, not by a failure of the held-out stochastic score. The same already frozen criteria and Python sample files were used; because the prior held-out result had already been observed, this is a retest rather than a fresh blinded holdout.

| Seed | Python genPPL | Strict Rust genPPL | Python entropy | Strict Rust entropy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16.572501 | 16.090782 | 3.966081 | 3.972742 |
| 1 | 15.659647 | 15.961892 | 3.964875 | 3.968466 |
| 2 | 16.402500 | 16.011160 | 3.976839 | 3.959642 |
| 3 | 16.351816 | 16.436652 | 3.973933 | 3.972631 |
| 42 | 16.176005 | 16.103382 | 3.959324 | 3.964234 |
| Mean | 16.232494 | 16.120773 | 3.968210 | 3.967543 |

The strict candidate again meets the fixed PPL and entropy criteria. Its result is `reports/phase4-tinystories-heldout-strict-v1/quality-v1.json`, SHA-256 `73cfb02e29c3ce9712205f06061bdfa7e5cb5f33d59fa43db85b6731067d1549`; the same evaluator qualification and batch-32 streaming caveats apply. Many same-seed sequences changed between Rust paths. A controlled first-level check on 1,024 draws found six different choices; both paths matched independent FP64 CDF choices for all draws, and the differences arose from changed logits, not a detected sampler discrepancy. Its raw report SHA-256 is `cda9faf33a2b87da1803e861e43d426bab666f8089727bf086f148689fc11c2b`.

## Sentinel-safe CUDA equivalence

A later safety fix bounds sampled indices before each subsequent embedding while retaining invalid sentinels for explicit rejection. For valid data this does not change the selected indices. The candidate regenerated all five strict-CUDA 256-sample files; each was byte-for-byte identical to its previously scored counterpart, including codes, IDs, decoded text and re-encoded tokens. The five-file comparison and candidate source/executable hashes are retained in `reports/phase4-safe-embedding-cuda-v1/equivalence.json`, SHA-256 `5f517656f8d39af87a8643c8aa017430fe82d44e87ac20fc89976c968b5f7d6e`. This supports reusing those CUDA quality scores for the safety candidate; Metal equivalence and updated performance remain separate checks.

The later shared Q/K/V split dispatch retains the existing CUDA implementation and was also checked against all five previously scored files: every file was byte-identical. That comparison and its source/executable hashes are in `reports/phase4-safe-split-cuda-v1/equivalence.json`, SHA-256 `2981686a46bd61873c3213e228b260edb1f263d5be7e455eff955e54af65d4bd`. The corresponding Metal split implementation remains a separate candidate until real-device validation.
