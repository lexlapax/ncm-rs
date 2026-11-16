# XSum Metal: original 32-passage shared histories

At commit `7b48c9f`, the [planned diagnostic](mlx-shared-history-plan-v1.md) forced Metal to follow the saved CUDA codes for all original 32 passages and compared each of 6,400 conditional decisions using the same random words. All 26,214,400 compared head logits were within the existing `atol=5e-4, rtol=2e-4` band; the maximum absolute difference was 0.0010986 and maximum per-position FP64-softmax total variation was 2.6033e-5. Metal sampling differed from the saved CUDA decision five times on identical CUDA logits and six times on forced-history Metal logits. The [full result](mlx-shared-history-v1.json) has SHA-256 `62d6997d5737c182ddbe9a2d47f8c0a5f2b9b9b011d354df7650a1129b60b30d`.

| Original first divergence (row, code position) | CUDA | Metal on same CUDA logits | Metal on forced-history Metal logits |
| --- | ---: | ---: | ---: |
| 14, 2 | 2301 | 2300 | 2300 |
| 7, 15 | 2802 | 2801 | 2801 |
| 0, 27 | 1305 | 1305 | 1304 |
| 25, 144 | 2164 | 2165 | 2165 |

Thus three original first differences reproduce with identical logits and words, while the fourth appears only on Metal's forced-history logits. These observations identify contributing effects at those positions, not the exact arithmetic cause or a universal bound. The five and six counts include other positions under forced CUDA histories; they are not the 284 free-generation code differences, which include cascades.

Linux full `tools/check.sh --cuda` and strict-CUDA diagnostic Clippy passed before the real-Metal probe. The retained 104,858,504-byte CUDA head trace has SHA-256 `c57f4f6c9a88e0f4d247dc7bc6fa6cb525e3c87945205fa33b0535af2de74018`; outputs and logs are on Mac under `reports/phase4-xsum-shared-history-mlx-v1/` and locally under `reports/phase4-xsum-shared-history-captured-v1/`. The [original exact failure](mlx-stochastic-real-v1.md) remains; investigate sampler accumulation precision without changing tolerances. Metal stochastic quality and performance remain unqualified.
