# Fused Metal interpolation: isolated result

The six predeclared fresh processes completed on Apple Silicon: three current
gather and three fused, rotating order. In all ten synthetic finite contiguous
shapes, every compared output bit matched the current gather implementation in
all six processes. This is not trained-model, nonfinite or strided-input parity.

| B / source / output / channels | Fused / gather process-median time | Gather / fused maximum peak active bytes |
| --- | ---: | ---: |
| 1 / 1 / 89 / 256 | 0.9351 | 397,088 / 200,480 |
| 1 / 2 / 89 / 256 | 0.9229 | 398,112 / 201,504 |
| 1 / 13 / 89 / 256 | 0.9202 | 409,376 / 212,768 |
| 1 / 50 / 89 / 256 | 0.9104 | 461,600 / 264,992 |
| 1 / 89 / 13 / 256 | 0.8685 | 151,968 / 125,344 |
| 4 / 13 / 89 / 256 | 0.8689 | 3,083,040 / 822,048 |
| 4 / 89 / 13 / 256 | 0.9070 | 901,536 / 508,320 |
| 2 / 37 / 67 / 33 | 0.8954 | 274,056 / 77,448 |
| 2 / 67 / 37 / 33 | 0.9246 | 112,096 / 53,488 |
| 1 / 5 / 1 / 33 | 0.9617 | 1,236 / 972 |

Each time ratio divides the median of the three process medians, with 30 warm
samples per process. These are diagnostic elapsed times including graph creation
and evaluation; the kernel handle was reused inside each process. Allocator
peaks were reset per shape and are not host RSS, physical GPU memory, or
full-model peaks. First measured B1 fused evaluations were 4.79/4.41/4.58 ms;
first gather evaluations were 23.32/19.98/19.52 ms. Each gather process ran
fused parity after its first measurement, so previous processes could warm
shared compiler caches; these numbers do not establish cold-start behavior.

One activity sample was collected during the very short run, with no Chrome or
WindowServer in its recorded top entries; that observation cannot establish GPU
exclusivity. Thermal checks before and after reported no warning. The initial
build failed at the workspace unsafe-code lint; the successful build used the
existing project pattern of a narrow allowance for the documented kernel call.
No GPU workload ran in the failed build. Both logs are retained.

This result justifies a narrowly isolated full-codec evaluation and checks for
nonfinite inputs, views, bounds, initialization, lifetime and startup before
considering production use. In particular, `MetalKernel` is not `Send` or `Sync`;
this prototype reuses a local handle but does not establish a production cache
or concurrent use. A model change, correctness qualification and both runtime
gates are still needed before adoption. Production behavior, pins and acceptance
criteria are unchanged; the separate steady-state live-memory issue stays open.

[Raw results and provenance](raw/mlx-linear-fused-poc-v1.tar.xz), SHA-256
`c837af7f0b56efae5b1cb971a307109f5a1cb04c756b59ead00ceeb8321a3710`.
The [offline reviewer](mlx-linear-fused-review-v1.py) checks all six report
hashes and parity observations; pass it the directory containing the extracted
`reports` directory.
