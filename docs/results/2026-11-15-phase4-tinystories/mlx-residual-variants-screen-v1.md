# Same-input Metal residual formulations

The [operator screen](mlx-residual-variants-plan-v1.md) compared the existing MLX 1D convolution, a height-one 2D convolution, and three matrix multiplies summed over kernel positions. Each used the same captured 128-row CUDA input, pinned weights and bias. Three warmups and five synchronized operator calls per formulation were measured in one process; these are exploratory operator timings without activity telemetry, not model-level or memory results.

| Formulation | Median ms | Convolution values outside diagnostic band | Blended values outside band | Blend max absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Existing 1D | 3.424 | 42 | 15,618 | 0.011719 |
| Height-one 2D | 2.903 | 42 | 15,618 | 0.011719 |
| Three matrix multiplies | 2.208 | 42 | 10,654 | 0.009277 |

The 2D and 1D blended captures were byte-identical. The three-matrix formulation reduced outside-band blended values by 31.8% on these 6,291,456 values and remained above zero exceptions. Its [same-input FP64 diagnostic](mlx-residual-fp64-v1.md) gave mean/p99/max absolute error `0.000006113 / 0.00008931 / 0.00703302`, compared with the existing MLX formulation's `0.000010428 / 0.00015296 / 0.01186914`. The new formulation exceeded the CUDA reference's FP64 error plus the existing band on 231 values, versus 2,270 for the existing MLX formulation. The pinned CUDA result itself exceeded the band versus FP64 on 7,697 values. These are diagnostic comparisons, not a new tolerance, quality gate, end-to-end performance result or claim that all numerical exceptions are resolved.

The summary `reports/phase4-mlx-residual-variants-v1.json` SHA-256 is `8d191648f1ca1f7c818cf28a2ad359b5ae4e47cfeedf0215fe59928a8fec7d6b`; the copied three-matrix capture `reports/phase4-mlx-residual-three-gemm-v1.safetensors` is `cf6c8eb5774b2a1338fa54efed0274be7884ba867756043d77c1f046a99a51a4`; FP64 result `reports/phase4-mlx-three-gemm-fp64-v1.json` is `5924501dcd0ed206c099bd6b60871c981dbfc0ad507d7a8684f5de8816430c91`. Input and source hashes remained stable; transferred files matched the Mac. The diagnostic's historical FP64 JSON key `rust` refers here to the Python MLX variant. The subsequent [Rust full-model candidate](mlx-first-residual-three-gemm-screen-v1.md) regressed numerical diagnostics and decisions and was rejected; no production change was made.
