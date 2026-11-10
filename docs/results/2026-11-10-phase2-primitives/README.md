# Phase 2 Linux shared components

Scope: native CPU/CUDA backend operations and shared codec components, not complete
free-running codec qualification. Linux performance/memory acceptance and all
Phase 2 Apple work remain pending. Existing pins, goldens and budgets are unchanged.

## Checks

```sh
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo test -p ncm-backend-cuda --features cuda --release \
  --test codec_components -- --nocapture
```

Run on the established RTX 3060 Ti/Linux environment, Rust 1.99.1 / Candle 0.11.1.
Full local checks pass: 24 base Rust tests, the additional real-CUDA component
suite, seven Python report tests, all 18 unchanged operator cases, fmt/Clippy/docs/
syntax. `tools/check.sh --cuda` now also runs CUDA-feature release tests, not just
the Phase 1 operator example. It still downloads no weights or starts SSH jobs.

The shared tiny fixture runner verifies every interned array's shape/dtype/byte
count/hash before loading. Both CPU and CUDA pass **140 floating comparisons /
75,348 elements** across batch-two and batch-one inputs. These comparisons include
isolated exact-erf GELU; all traced decoder-block submodules; level-zero normalized
cascades; area/linear sampling; codebook distances/lookup; Phi convolution; and
cumulative additions. Quantization uses reference residual inputs at this stage,
so this is **component parity, not complete free-running encode parity**.

| Backend | Largest absolute error across comparisons |
| --- | ---: |
| CPU | 1.1444091796875e-5 |
| CUDA | 7.62939453125e-6 |

Budgets remain the predeclared primitive/component/distance/accumulated classes in
[PHASE2.md](../../PHASE2.md). Every tiny level's selected code IDs matches exactly.
Additional deterministic tests cover known-margin codebook selection, first-index
argmin/argmax ties (including signed zero and a 1,025-wide row), independent retained
snapshots, dtype/device mismatch rejection and malformed convolution arguments.

Raw release output: `raw/components.txt`. Source is the accompanying commit; the
fixed fixture originated in `97a3125`. These tests are not timing benchmarks.
Affected-workload performance and memory comparisons remain required at the Linux
gate; no efficiency or trained-model reconstruction claim is made here.

## Negative findings and explicit adapter fixes

- The original Candle CPU batched transposed-convolution path produced incorrect
  second-batch values for a `[2,16,4]` BCL input and diagonal `[16,16,4]` IOK kernel,
  stride four. For example, the first output in batch two was 1 rather than 64 in
  a deterministic increasing-input test. The normalized tiny cascade exposed this
  as a ~1.9974 error. Single-batch native calls pass. The CPU adapter now dispatches
  batch slices separately and concatenates them; CUDA still uses native batched
  convolution and passes the same regression. No native dependency patch or host
  fallback for GPU tensors was introduced. This CPU workaround is not a speedup.
- Candle CUDA argmin selected tied index 1,024 instead of first index 3 in the new
  test. Selection now first reduces values, then minimizes indices masked by exact
  equality with the extremum, entirely on-device. Both argmin and argmax use this
  rule. No epsilon perturbation of scores, tolerance change or silent tie waiver.
- The retained original-Python CPU/CUDA trained-reference discrepancy is separate
  from these adapter defects and remains documented in the reference report.

The shared decoder block explicitly uses bidirectional attention, Q/K RMSNorm and
exact GELU; no generator logic or Phase 3 implementation is included.
