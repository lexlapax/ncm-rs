# XSum Metal stochastic capture progress

The [comparison plan](mlx-stochastic-quality-plan-v1.md) had SHA-256
`b1c47cdab03cd92384573e1a7dd317a1ee601e70b76b0ff5176d3b397486560c` before
capture; it reuses the existing pinned follow-up protocol and changes no margin.
Seed 4's 2,048-row Metal capture completed and copied files matched the Mac by
SHA-256; all 64 input/receipt pairs passed the new scorer's structural validation.
Its first batch was byte-identical to a separate seed-4 smoke run.

As expected for independent RNG streams, a source-checked diagnostic found
91,622 differing raw IDs across all 2,048 rows versus the retained Python seed-4
capture (`reports/phase4-xsum-mlx-stochastic-seed-4-diagnostic-v1.json`). This
is not a quality failure or a statistical comparison. The other seeds, evaluator
replay, scoring and bootstrap remain pending. The original CUDA stochastic
failure and controlled-draw differences remain recorded.

The new Metal-aware scorer exactly replayed the retained Python seed-4 score on
all 2,048 rows and aggregate metrics, on the same GPU model with driver 620.70.06.
Replay result `reports/phase4-xsum-mlx-stochastic-scorer-replay-v1.json` has
SHA-256 `b46b1aa7b08da9c54ad13f7f8709a8ea32914b4bfecb3dc8dde3bd886ab4524c`.

Seeds 4–7 have since completed, were copied and per-file hash-verified, and
were scored on CUDA with the replay-qualified wrapper. Each contains 2,048 rows.
The score-file SHA-256 values in seed order are
`a2e54a1e29796e970e4c06ee6b9d308014fff959b53641fc8c6820196b4a0c5f`,
`dafbc9f9a5596b527cea04eea94303a3167f12624d7248093d458175f0c55b32`,
`9cd8effa4f2c3c989768b098d6df5fd0fcae12fb592c0cf01ddf3ddd0780bc65`, and
`21f70ff2d4467f4b0e37d8b487230276b0a103e42c0187c3d0cc99198aa42cc8`.
Seed 8 and the predeclared five-seed bootstrap remain pending; individual seed
scores do not establish the quality rule.

Before the final comparison, the scorer wrapper was strengthened to record and
recheck hashes of its imported scoring helpers. The protocol, evaluator routines,
metrics and margins were not changed. A fresh replay again matched the retained
Python seed-4 rows and metrics exactly (`reports/phase4-xsum-mlx-stochastic-scorer-replay-v2.json`,
SHA-256 `3483eda3a21df50417c3fc959a5544f803ff89b316d740c9ba26dd5f5b092587`).
Seeds 4–7 were rescored and every per-row score and aggregate metric matched their
retained v1 score files exactly. The verification result is
`reports/phase4-xsum-mlx-stochastic-scorer-provenance-check-v1.json`, SHA-256
`ec64c62f05212742b0ae1944c729785d944d47c847c444d2f192644db781e214`.
The final seed and frozen five-seed bootstrap are still pending.

Seed 8 subsequently completed and the full predeclared comparison was analyzed;
see [the result](mlx-stochastic-followup-v1.md). The earlier pending statements
above describe the intermediate state and do not override that result.
