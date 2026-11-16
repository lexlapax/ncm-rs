# Generator outlier positions across CUDA and Metal

The [targeted diagnostic](mlx-outlier-overlap-plan-v1.md) compared Rust MLX and independently retained Python MLX generator logits against the same pinned Python CUDA tensors. For each of five seeds, it used 16 identical forced-code rows and 12 logit tensors, covering 293,601,280 values. Both comparisons use the unchanged accumulated band `atol=5e-4, rtol=2e-4` relative to CUDA.

| Seed | Both outside | Rust only | Python MLX only | Neither outside |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 148 | 2 | 0 | 58,720,106 |
| 101 | 513 | 9 | 4 | 58,719,730 |
| 102 | 161 | 4 | 1 | 58,720,090 |
| 103 | 137 | 1 | 3 | 58,720,115 |
| 104 | 154 | 2 | 3 | 58,720,097 |
| **Total** | **1,113** | **18** | **11** | **293,600,138** |

Thus 1,113 of Rust's 1,131 and 1,113 of Python MLX's 1,124 outside-band positions overlap. Every tensor's marginal counts exactly match the earlier independently obtained Rust/CUDA and Python-MLX/CUDA results, and input codes and codec IDs matched byte-for-byte before execution. This is strong evidence of a shared cross-runtime or adaptation contribution for these inputs, not proof of its precise operation or a pass of the CUDA-relative band. Rust and Python MLX differed in FP32 bits at 271,571,348 positions, so shared outlier positions do not mean identical calculations. Existing checked argmax and quality evidence retains its own scope; these results do not explain the latency tail or justify changing a numerical tolerance.

The Metal probe was built from commit `949b75e` in a new isolated source copy, with the reviewed patch, existing prebuilt native MLX and pinned artifact. The first build attempt failed because moving the original example broke a manifest entry; restoring the original in the isolated copy allowed the new example to build and run. Workspace crate and vendored source, binary, native libraries, artifact, and input trace fingerprints matched before and after each run; external Cargo dependency source was not separately fingerprinted. The one-seed result `reports/phase4-outlier-overlap-100-v1/result.json` has SHA-256 `076a49d80afc4d46bb943a4abb7242fdee5bf7185bc9423fdd689587f2562c5b`; its fingerprint manifest is `175f50ac835d5e9841386c533e82dea2808dd7bddc9ac504352dac24e8edcd04`.

The expanded result files for seeds 101–104, respectively, have SHA-256 `3d1ac8417bb1ff8a2700b06eb085868ad1cef676ea564069673061a734457496`, `390a7faefb0c50e2b7bb528dec39d4b2547707c6a3632f9cbc43d570bed42948`, `07273e30579f28ea4cc5651b130a509e68d4a977d934282b620a320ad5adbac8`, and `b6807e34ac5713b43352e9b80021b85069fd7cc15c6477bbd53882777c8d9950`; their fingerprint manifest is `b8cc37e6c5f4f79b49d97849b7d5f61c3ce2929302c981c953aec331457cf745`. Retained copies are under ignored `reports/phase4-outlier-overlap-{100,expanded}-v1/`, and matched the Mac hashes. The marginal checks additionally used the prior per-seed summaries linked from the [direct Python comparison](mlx-python-cross-runtime-v1.md) and [Rust screen](mlx-generator-qk-rope-portable-screen-v2.md). This is a diagnostic of retained samples, not a new full-model acceptance gate.
