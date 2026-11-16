# Same-input MLX and CUDA residual versus FP64

The [same-input stage probe](mlx-cascade-stages-v1.md) localized an early MLX/CUDA mismatch to the first reconstruction residual convolution. The exact same 128-row input and artifact weights were used to compute an FP64 convolution and blend on CUDA, then each runtime's FP32 blended result was compared with it. The FP64 calculation is a diagnostic reference, not a replacement oracle or parity gate.

| Measure | Python CUDA FP32 | Python MLX FP32 |
| --- | ---: | ---: |
| Mean absolute error vs FP64 | 0.0000128490 | 0.0000104280 |
| p99 absolute error vs FP64 | 0.00020443 | 0.00015296 |
| Maximum absolute error vs FP64 | 0.00473058 | 0.01186914 |
| Values closer to FP64 | 2,088,302 | 3,988,751 |

There were 214,403 equal-error values among 6,291,456. CUDA itself lay outside the existing diagnostic band versus FP64 on 7,697 values; MLX exceeded CUDA's absolute FP64 error plus that band on 2,270. Thus higher precision alone cannot be assumed to reproduce CUDA or remove all existing CUDA-relative exceptions. This does not identify the exact reduction ordering responsible, justify a tolerance change, or establish end-to-end numerical parity.

The unchanged diagnostic `tools/reference/tinystories_residual_fp64.py` uses historical JSON keys `python` and `rust`; in this run those mean Python CUDA and Python MLX, respectively. Input `cascade.3` arrays were verified exactly equal before running. CUDA two-tensor reference SHA-256 `582da9f8a2d6c84375e236876305f3fed0d6eb7f39e3aada645eee76e9f18426`; MLX capture `d1d25bb295d296c3a68e5cc85e298e5945f4311eb7a2288cbf62ce229eeb980f`; output `reports/phase4-mlx-residual-fp64-v1.json` `32a6ae75153637074aa7cf80fa07dab900d8f198d45dfed418c7c7493301c30f`. The Mac's temporary transferred CUDA trace was removed after capture; the original retained Linux trace and both derived evidence files remain.
