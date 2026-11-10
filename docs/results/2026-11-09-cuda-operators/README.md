# Phase 1: Linux/CUDA operator capability report

**Status: measured operator milestone, not a complete Phase 1 or NCM implementation.**
18 synthetic operator cases pass the fixed FP32 tolerance against a PyTorch CPU
oracle on both Candle CPU and Candle CUDA. No checkpoint, generated text, task
quality, throughput, peak-memory or MLX result is claimed.

## Environment and provenance

- NVIDIA RTX 3060 Ti, 8 GiB; Linux, driver 615.78.08.
- Rust 1.99.1, Candle **0.11.1**, CUDA toolkit 13.4; compute capability 86;
  host compiler `/usr/bin/g++-15`. Release profile with thin LTO.
- PyTorch **2.15.0+cu132**, Python 3.14.9; full Python package versions are in
  [`requirements-linux.lock`](../../../tools/reference/requirements-linux.lock).
- FP32, TF32 disabled, eager/math attention baseline. Synthetic inputs use a fixed
  formula, not backend RNG. The FP16 roundtrip case uses non-binary fractions to
  test actual rounding. Tolerance was set before testing: `atol=1e-5, rtol=1e-4`.
- Upstream semantic reference: `c70274273f489abc73832b2b98584a8483b6b203`.
  The oracle composes PyTorch operators matching inspected NCM conventions; it
  **does not execute upstream model classes** or establish checkpoint compatibility.
- Reports were collected from the candidate tree based on commit `0d05682`.
  Each Rust raw report embeds SHA-256 hashes of the compiled probe, helper, backend
  sources, manifest and Cargo.lock. Python reports embed the oracle script hash.
  These hashes identify the candidate exactly without a self-referential commit ID.
  [`environment.json`](environment.json) records tools, build flags, fixture/source
  digests, collection time and execution order.

## Method

Three fresh-process runs per runtime, alternated Python/Rust, then Rust/Python,
then Python/Rust. Each case has five warmups and thirty timed evaluations per
process (90 pooled measurements). Timings include host dispatch and device
synchronization; input upload and output readback are excluded. Outputs are checked
outside the timed region. Native operations are recomputed for every trial.

These are deliberately **tiny, launch-bound capability fixtures**, not representative
NCM benchmarks. Different backend primitives and CUDA library versions prevent
attribution to language alone. Clocks/power were not locked and the GPU also serves
a desktop: treat close differences as inconclusive. No speedup target or regression
budget is established by this measurement.

## Results

Times below are pooled **median / p95, microseconds**. Error is maximum absolute
Candle CUDA error versus the PyTorch CPU fixture across all three runs. All outputs
are finite and every element passes the absolute-plus-relative tolerance; Python
CUDA also passes all 18 fixtures.

| Case | Candle CUDA µs | PyTorch CUDA µs | Max absolute error |
| --- | ---: | ---: | ---: |
| matmul_asymmetric | 8.94 / 10.32 | 9.37 / 11.44 | 0 |
| conv1d_layout | 24.56 / 29.10 | 20.06 / 21.33 | 0 |
| conv_transpose1d_layout | 17.23 / 18.34 | 20.14 / 22.86 | 0 |
| conv_transpose1d_prime89 | 15.07 / 17.60 | 25.50 / 26.83 | 0 |
| group_norm_sequence_axes | 41.62 / 44.66 | 21.99 / 26.86 | 4.77e-7 |
| layer_norm | 32.13 / 34.00 | 10.35 / 10.89 | 1.19e-7 |
| rms_norm | 21.99 / 23.62 | 21.07 / 21.97 | 1.19e-7 |
| area1d_7_to_4 | 40.03 / 41.69 | 10.36 / 10.73 | 1.49e-8 |
| area1d_89_to_13 | 161.51 / 163.62 | 14.18 / 14.52 | 2.98e-8 |
| area1d_1_to_1 | 9.66 / 10.85 | 10.27 / 10.56 | 0 |
| area1d_3_to_7 | 62.58 / 64.27 | 10.81 / 11.24 | 0 |
| linear1d_3_to_7 | 94.75 / 95.98 | 12.98 / 13.58 | 1.19e-7 |
| linear1d_1_to_9 | 77.54 / 80.73 | 12.97 / 13.42 | 2.98e-8 |
| linear1d_7_to_3 | 43.19 / 44.53 | 13.06 / 13.80 | 4.17e-7 |
| linear1d_7_to_7 | 94.71 / 96.23 | 12.96 / 13.41 | 0 |
| prefix_attention_padded | 40.12 / 41.21 | 53.42 / 59.74 | 2.98e-8 |
| rope_split_half_level_resets | 25.31 / 27.23 | 22.03 / 22.91 | 0 |
| f16_roundtrip | 7.73 / 8.18 | 10.41 / 10.91 | 0 |

[`summary.json`](summary.json) includes per-process median ranges. The six small
raw reports are retained permanently in [`raw/`](raw/) with individual trials,
first-execution timings, errors and hashes; no expiring CI asset is required.

## Findings and limitations

- Candle CUDA can execute the required starter primitives on this Ampere GPU.
  Native cuDNN/FlashAttention dependencies are not needed for this correctness spike.
- Conv/transposed-conv channel layouts, prime stride 89, group-normalization axes,
  half-pixel interpolation, overlapping area bins, prefix masking and per-level
  split-half RoPE have concrete regression fixtures now.
- Correctness-first interpolation launches many tiny operations and is substantially
  slower than PyTorch's specialized kernels. This is a negative performance result,
  retained intentionally. Fuse or vectorize later, after broader shape/codec parity.
- The new code is a candidate operator layer, not a frozen cross-runtime abstraction.
  No sampler, codebook search, tied heads, weight loader, full codec/generator,
  mixed-precision policy, training or model-quality suite exists yet.
- CPU tests cover these cases plus malformed fixtures, nonfinite comparisons,
  invalid operator arguments and all-masked-row rejection. They are not exhaustive
  operator validation or substitutes for real-model tests.
- **Phase 1 remains open:** trusted checkpoint conversion, artifact roundtrip,
  tokenizer parity and model-level oracle traces still need Linux validation.
  macOS/MLX follows the Linux portion of this phase, not all later CUDA phases.
- `macstudio` is reachable (M1 Ultra, 64 GiB), but has competing workloads.
  **Notify the user and coordinate availability before starting MLX workloads.**
  Only read-only environment inspection has occurred there.

## Reproduce

From the repository root (Python 3.14 on Linux):

```sh
python3 -m venv .reference/venv
.reference/venv/bin/python -m pip install -r tools/reference/requirements-linux.lock
cargo test -p ncm-backend-cuda --features cpu --locked
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo build -p ncm-backend-cuda --features cuda --example probe --release --locked
mkdir -p reports/reproduction
# Repeat in three NEW processes per runtime, alternating their order.
.reference/venv/bin/python tools/reference/operators.py run \
  --fixture tests/fixtures/operators-v1.json --device cuda \
  --output reports/reproduction/trial-1-python.json
target/release/examples/probe cuda tests/fixtures/operators-v1.json \
  reports/reproduction/trial-1-rust.json
# After also collecting trial-2-* and trial-3-*:
python3 tools/bench/compare.py --output reports/reproduction/summary.json \
  reports/reproduction/trial-*.json
```

Report writers refuse to overwrite existing files. Change host compiler and compute
capability appropriately for another machine and label that as a different environment.
To review/regenerate the oracle, use `operators.py generate --fixture <NEW-path>`;
never silently replace the checked-in fixture to make a regression pass.
