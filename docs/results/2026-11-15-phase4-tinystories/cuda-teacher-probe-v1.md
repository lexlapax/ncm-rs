# TinyStories initial CUDA teacher-forced probe

This is an exploratory comparison, not a numerical parity gate, stochastic generation test, quality evaluation or performance measurement. The pinned upstream Python implementation and Rust both ran on an NVIDIA GeForce RTX 3060 Ti, using two synthetic sequences and identical fixed code indices. Upstream used FP32, TF32 disabled and math SDPA. The Rust probe compares each head against those forced inputs; it does not choose its own codes. Artifact and source identities are in the [Phase 4 contract](../../PHASE4-TEXT.md).

Python trace: `reports/phase4-tinystories-cuda-reference-v1/trace.safetensors`, SHA-256 `cd1ed6bd07186f3a3c460b64db50df0a586f00d8400f7966e3818d9223b16d66`. Rust diagnostic output: `reports/phase4-tinystories-rust-probe-v2.json`, SHA-256 `902f7b6691748f8f2e94a00caf50d34a1d00f8f57104373bc27c1b6933adda5b`. These raw files are local and ignored by Git.

| Generator level | Logit maximum absolute difference | Logit RMS difference | Different argmax positions |
| ---: | ---: | ---: | ---: |
| 1 | 0.000793457031 | 2.5582996e-05 | 0 |
| 2 | 0.000553131104 | 1.92696383e-05 | 0 |
| 3 | 0.00272750854 | 3.98453838e-05 | 0 |
| 4 | 0.00085735321 | 2.809847e-05 | 0 |
| 5 | 0.00958251953 | 4.12371683e-05 | 0 |
| 6 | 0.00290679932 | 4.70096629e-05 | 0 |
| 7 | 0.00188446045 | 4.81594656e-05 | 0 |
| 8 | 0.00378417969 | 6.27253989e-05 | 0 |
| 9 | 0.00988769531 | 0.000107803012 | 0 |
| 10 | 0.00695800781 | 9.74225826e-05 | 0 |
| 11 | 0.0031747818 | 7.62893238e-05 | 0 |

Final decoder logits: maximum absolute difference 0.0016374588, RMS 0.000142430199; 0 of 256 final IDs differ. First-level reconstruction maximum absolute difference 0.00341796875, RMS 5.46814267e-05. The differences need a declared numerical budget and further diagnosis; agreements on these two inputs do not imply stochastic or dataset parity.

Reproduce with `tools/reference/tinystories_trace.py`, using its required pinned upstream and source arguments, and `tinystories_teacher_probe` built with the CUDA feature. The trace overrides sampling with fixed indices explicitly; no random-seed equivalence is asserted. Existing Phase 3 Sudoku CUDA synthetic golden passed separately after the shared-transformer refactor. Padded TinyStories prefixes are rejected by the prototype; sampling, full quality, timing, memory and all Apple validation remain pending.

A separate expanded probe on the same two synthetic inputs captured encoder and decoder outputs (`reports/phase4-tinystories-rust-probe-v3.json`, SHA-256 `bf6765fbba0ff6867652c595b926277ab25ac42d5cf6abd86ac7f93050aaa6f2`). The pre-quantization maximum absolute difference was `4.7684e-7`; all 1,282 encoded codes agreed. Decoder logits from those codes had maximum absolute difference `0.00225544` and RMS `0.000158897`. This is still a small synthetic diagnostic, not a codec parity gate.
