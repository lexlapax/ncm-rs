# LM1B-medium and LM1B-large on Metal

All four items of the [lean bar](user-decisions-v1.md#3-minimum-bar-for-lm1b-medium-and-lm1b-large) pass on
the Mac Studio (M1 Ultra, 64 GiB) at commit `30349ae`, 2026-11-19. The MLX build, `tools/check.sh --mlx`,
the five real-Metal sampler tests and feature Clippy all pass. The medium and large inference artifacts were
copied from Linux, and every file hash matches.

## Teacher-forced parity

The MLX `lm1b_small_teacher_trace` forces the same two development-code histories as the
[CUDA comparison](medium-large-cuda-parity-v1.md). It is compared with the **strict** Python CUDA reference
(math attention, FP32 matmul, convolution TF32 off) using the unchanged `compare_lm1b_small_teacher_trace.py`
policy.

| Size | Tensors within policy | Level argmax mismatches | Greedy / penalized ID mismatches | Max level-logit error |
| --- | ---: | ---: | --- | ---: |
| medium | 51 / 51 | 0 of 11 levels | 0 / 0 | 6.71e-4 |
| large | 51 / 51 | 0 of 11 levels | 0 / 0 | 1.10e-3 |

The level-logit errors are within the accumulated policy's combined absolute and relative bound.

| Item | SHA-256 |
| --- | --- |
| Medium Metal trace | `6b25cccbf8fd44834e69b1bbb33ebd3036feea5487287d6f1509ef5e803b0102` |
| Medium comparison | `7b584ad16b9aefa2e0d7a52dc03c220f663a7cea59b8b1a8feeab43f155f5ddf` |
| Large Metal trace | `4bf426ce6a05ef2d32af44bc02feff46140047376502a866aa29066770aa77c5` |
| Large comparison | `1068456b6c18373a630b548ec8a4ac1a60afec38eb53f4d90d4701399929472a` |

The reference traces are the strict ones recorded in the CUDA parity report (`5e5a2e91…`, `4c49bcd7…`).

## CLI generation

`ncm sample-lm1b ARTIFACT mlx INPUT SEED ROTARY --decode` (build feature `lm1b-mlx-experimental`) generates
two rows at seed 7 for small, medium and large. Each row has 641 codes and 128 IDs, and the output includes
decoded text. Peak process RSS was 1.6, 3.3 and 6.3 GiB.

## Quality

The 80-seed Metal campaigns pass at both sizes: medium perplexity +0.17 [-0.39, 0.71], large +0.10
[-0.39, 0.60], and both entropy intervals are inside ±0.0046. See
[the campaign result](cohort-campaign-v1.md#prospective-metal-results-medium-and-large).

## Latency and memory against Python MLX

The protocol is the same as the [small Metal timing](small-metal-timing-v1.md): `run_lm1b_small_mlx_timing.py --size SIZE`,
three fresh processes per runtime and batch, 5 warmups and 30 measured requests each, with peak RSS from
`/usr/bin/time -l`. Python is the qualified MLX reference forward pass with on-device categorical sampling.
**Every cell passes median, p95 and RSS.**

| Size | Batch | Python median / p95 (s) | Rust median / p95 (s) | Rust faster (median) | Python / Rust RSS (MiB) | Python / Rust MLX peak (MiB) |
| --- | ---: | --- | --- | ---: | --- | --- |
| medium | 1 | 0.2250 / 0.2260 | 0.2095 / 0.2106 | 6.9% | 3,366 / 3,327 | 1,832 / 1,806 |
| medium | 4 | 0.4878 / 0.4918 | 0.4524 / 0.4563 | 7.2% | 3,366 / 3,328 | 2,355 / 2,247 |
| medium | 16 | 1.4294 / 1.5076 | 1.1470 / 1.1710 | 19.8% | 3,367 / 3,362 | 3,411 / 3,258 |
| large | 1 | 0.3869 / 0.3938 | 0.3640 / 0.3705 | 5.9% | 6,494 / 6,456 | 3,448 / 3,415 |
| large | 4 | 0.8756 / 0.8890 | 0.8026 / 0.8120 | 8.3% | 6,495 / 6,455 | 4,147 / 3,990 |
| large | 16 | 2.6630 / 6.3104 | 2.2464 / 3.3072 | 15.6% | 6,495 / 6,451 | 4,970 / 4,683 |

The RSS margins are small (0.15–1.2%), because both processes' peaks are set mostly by model loading. Large
batch-16 p95 is noisy for both runtimes, with occasional slow requests in each; the medians are stable.
Random streams differ, so outputs are not compared here.

| Item | SHA-256 |
| --- | --- |
| Medium `result.json` | `ec16c941642a80c376103eb0ad04ec12413721f0944dbcc59b28320e6d32cf32` |
| Large `result.json` | `98e27d72309c83e0c78be2fc42e9df0b3d8846f27298d14daa99b3ac6ad9c5d4` |

Raw outputs are in ignored `reports/phase4-lm1b-{medium,large}-mlx-timing-v2/`,
`reports/phase4-lm1b-{medium,large}-teacher-metal-v1/` and `reports/phase4-lm1b-*-metal-cli-v1.json`, on the Mac and Linux.

## Reproduce (on the Mac, from the repository root, with the MLX build environment in the context map)

```sh
cargo build --locked --release -p ncm-backend-mlx --features mlx,lm1b-experimental \
  --example lm1b_small_teacher_trace --example lm1b_small_mlx_timing_probe --example lm1b_small_cohort_campaign
cargo build --locked --release -p ncm-cli --features lm1b-mlx-experimental
target/release/examples/lm1b_small_teacher_trace artifacts/canonical/lm1b-medium-inference-v1 \
  reports/phase4-lm1b-medium-teacher-cuda-reference-v1 reports/NEW-TRACE
.reference/mlx-venv/bin/python tools/bench/run_lm1b_small_mlx_timing.py --size medium \
  --rotary reports/phase4-lm1b-metal-cli-rope-v1.safetensors --output reports/NEW-TIMING
```

Compare traces on Linux with `compare_lm1b_small_teacher_trace.py`, as in the CUDA parity report.
