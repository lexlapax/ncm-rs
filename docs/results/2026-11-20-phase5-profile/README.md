# Phase 5 step 2: stage-breakdown profiling (CUDA and Metal)

2026-11-20. At commit `51046ab`, `phase5_stage_profile` measured every released profile at batch 1
and its largest measured batch in fresh processes on both target GPUs:

- **Linux / CUDA:** NVIDIA GeForce RTX 3060 Ti (8 GiB, driver `620.73.04`, `llama-server.service` stopped; three desktop GPU clients remained). Summary: [cuda-profile-summary-v1.json](cuda-profile-summary-v1.json) (SHA-256 `7941fd98ade8f37199862796d4660fc71a06b65969867d87c784be4df15996bc`).
- **macOS / Metal:** Apple M1 Ultra (64 GiB, macOS 26.7.3, native MLX 0.32.2, `dotsetup.rmlx` idle/untouched). Summary: [mlx-profile-summary-v1.json](mlx-profile-summary-v1.json) (SHA-256 `0514461ed7ecb61dcb0ad600c62d3b0dce642d15b8873628d21dfd1cb8f4c10a`).

Each fresh process recorded: (1) cold model load split into `Artifact::load*` vs backend weight upload,
(2) warm uninstrumented latency (3 warmups, 10 synchronized trials), (3) warm stage-synchronized
breakdown (1 warmup, 5 trials; CUDA stream sync / MLX tensor `eval` at each stage boundary without
host tensor copies), and (4) a separate post-warmup memory trace capturing process `VmRSS` / `VmHWM`
and device memory counters (`cuda_default_pool_bytes` on CUDA; `mlx::memory::{active,cache,peak}` on
Metal). Inter-stage synchronization adds small barrier overhead vs uninstrumented warm latency
(reported side by side below).

## Linux / CUDA stage breakdown

| Profile | Batch | Model load (ms) `art + upload` | Warm uninst. med / p95 (ms) | Prefix (ms) | Generator levels sum (ms) | Top level (ms) | Sampling + penalties (ms) | Codec decode (ms) | Host readback (ms) | Peak RSS HWM (MiB) | CUDA pool peak used / res (MiB) |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `sudoku-easy` | 1 | 217.4 (`174.6 + 42.8`) | 26.18 / 26.52 | 0.03 | 25.30 | L11: 2.6 | 0.33 + 0.00 | 0.23 | 0.03 | 604 | 169 / 192 |
| `sudoku-easy` | 128 | 216.3 (`174.2 + 42.1`) | 1628.43 / 1632.02 | 0.44 | 1788.80 | L11: 238.5 | 1.08 + 0.00 | 9.80 | 2.33 | 604 | 1285 / 1440 |
| `tinystories` | 1 | 738.0 (`615.3 + 122.8`) | 41.64 / 42.06 | 0.03 | 40.80 | L11: 5.9 | 0.52 + 0.00 | 0.72 | 0.03 | 941 | 493 / 512 |
| `tinystories` | 128 | 734.0 (`615.8 + 118.2`) | 2361.55 / 2364.80 | 0.20 | 2312.37 | L11: 472.0 | 7.71 + 0.00 | 85.17 | 3.10 | 934 | 1808 / 1952 |
| `xsum` | 1 | 1024.3 (`893.7 + 130.6`) | 157.50 / 159.40 | 0.07 | 162.26 | L06: 18.4 | 0.18 + 1.07 | 0.77 | 0.01 | 1074 | 633 / 672 |
| `xsum` | 32 | 852.0 (`721.1 + 130.9`) | 4366.58 / 4399.01 | 1.67 | 4663.82 | L08: 545.0 | 0.48 + 25.90 | 8.53 | 0.02 | 1074 | 2201 / 2848 |
| `wmt14` | 1 | 1469.3 (`1252.3 + 217.0`) | 59.02 / 59.89 | 0.03 | 57.55 | L10: 6.3 | 0.20 + 0.00 | 1.46 | 0.01 | 1517 | 1050 / 1088 |
| `wmt14` | 32 | 1473.9 (`1255.1 + 218.7`) | 1206.91 / 1212.90 | 0.27 | 1225.07 | L10: 147.8 | 0.70 + 0.00 | 30.30 | 0.03 | 1520 | 2204 / 2624 |
| `lm1b-small` | 1 | 1468.5 (`1295.5 + 173.0`) | 43.01 / 43.80 | 0.05 | 37.24 | L11: 5.9 | 1.69 + 2.03 | 2.02 | 0.02 | 1241 | 839 / 864 |
| `lm1b-small` | 16 | 1222.4 (`1050.8 + 171.6`) | 374.95 / 380.42 | 0.23 | 302.27 | L11: 59.1 | 25.92 + 16.96 | 32.44 | 6.31 | 1243 | 1350 / 1536 |
| `lm1b-medium` | 1 | 2836.4 (`2487.2 + 349.2`) | 119.51 / 120.51 | 0.05 | 113.57 | L11: 18.5 | 1.77 + 2.18 | 2.15 | 0.02 | 2078 | 1676 / 1696 |
| `lm1b-medium` | 16 | 2333.7 (`1985.5 + 348.2`) | 1061.99 / 1067.74 | 0.24 | 985.79 | L11: 196.7 | 28.30 + 16.88 | 32.55 | 5.61 | 2078 | 2187 / 2336 |
| `lm1b-large` | 1 | 5344.5 (`4667.6 + 676.9`) | 248.62 / 252.00 | 0.06 | 242.45 | L11: 39.3 | 1.80 + 2.24 | 2.21 | 0.02 | 3636 | 3233 / 3264 |
| `lm1b-large` | 8 | 4427.8 (`3754.2 + 673.6`) | 1170.13 / 1178.90 | 0.14 | 1132.08 | L11: 215.0 | 14.70 + 9.57 | 16.24 | 3.47 | 3636 | 3472 / 3520 |

## Apple Silicon / Metal stage breakdown

| Profile | Batch | Model load (ms) `art + upload` | Warm uninst. med / p95 (ms) | Prefix (ms) | Generator levels sum (ms) | Top level (ms) | Sampling + penalties (ms) | Codec decode (ms) | Host readback (ms) | RSS art → up → HWM (MiB) | MLX peak (MiB) |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| `sudoku-easy` | 1 | 198.0 (`175.5 + 22.5`) | 54.36 / 54.81 | 0.37 | 60.83 | L11: 5.8 | 3.69 + 0.00 | 0.95 | 0.36 | 174 → 338 → 359 | 262 |
| `sudoku-easy` | 128 | 187.4 (`165.4 + 22.0`) | 1263.80 / 1284.74 | 1.20 | 1537.03 | L09: 178.8 | 5.11 + 0.00 | 7.57 | 0.63 | 175 → 339 → 357 | 2618 |
| `tinystories` | 1 | 703.3 (`617.2 + 86.1`) | 93.67 / 94.77 | 0.37 | 95.99 | L11: 11.0 | 15.16 + 0.00 | 1.60 | 0.14 | 501 → 992 → 1016 | 607 |
| `tinystories` | 128 | 668.8 (`583.8 + 85.0`) | 1631.91 / 1638.94 | 1.12 | 1552.36 | L11: 300.8 | 21.57 + 0.00 | 35.69 | 0.23 | 501 → 992 → 1014 | 2475 |
| `xsum` | 1 | 1308.4 (`1225.0 + 83.4`) | 212.74 / 214.08 | 1.18 | 209.50 | L08: 25.2 | 6.11 + 8.07 | 2.09 | 0.22 | 965 → 1546 → 1585 | 1118 |
| `xsum` | 32 | 1269.0 (`1186.7 + 82.3`) | 2920.73 / 2939.75 | 2.46 | 2926.06 | L08: 341.9 | 4.54 + 49.95 | 34.01 | 0.72 | 965 → 1546 → 1639 | 2583 |
| `wmt14` | 1 | 2008.4 (`1277.6 + 730.9`) | 97.32 / 98.58 | 0.45 | 104.58 | L08: 10.6 | 5.81 + 0.00 | 2.38 | 0.00 | 643 → 1103 → 1562 | 1163 |
| `wmt14` | 32 | 1968.5 (`1233.6 + 734.9`) | 819.79 / 822.12 | 0.52 | 790.49 | L10: 88.9 | 5.64 + 0.00 | 30.49 | 0.00 | 643 → 1103 → 1561 | 2835 |
| `lm1b-small` | 1 | 1124.6 (`1044.8 + 79.8`) | 102.02 / 102.88 | 0.40 | 107.96 | L01: 12.9 | 5.98 + 17.41 | 3.14 | 0.09 | 822 → 1620 → 1674 | 927 |
| `lm1b-small` | 16 | 1075.2 (`996.2 + 79.0`) | 473.78 / 480.93 | 0.73 | 378.00 | L01: 116.1 | 10.73 + 82.33 | 10.61 | 0.07 | 822 → 1620 → 1673 | 2172 |
| `lm1b-medium` | 1 | 3932.2 (`3774.3 + 157.8`) | 209.22 / 211.79 | 0.41 | 206.24 | L11: 25.6 | 6.14 + 17.44 | 2.82 | 0.06 | 1659 → 3295 → 3328 | 1806 |
| `lm1b-medium` | 16 | 3667.6 (`3497.2 + 170.4`) | 1053.27 / 1071.40 | 0.74 | 959.86 | L11: 149.7 | 12.35 + 82.43 | 10.66 | 0.07 | 1659 → 3295 → 3358 | 3258 |
| `lm1b-large` | 1 | 3987.0 (`3681.7 + 305.3`) | 384.00 / 386.65 | 0.47 | 368.92 | L11: 49.7 | 6.66 + 24.45 | 3.10 | 0.11 | 3217 → 6410 → 6455 | 3415 |
| `lm1b-large` | 8 | 3836.9 (`3530.8 + 306.1`) | 1298.73 / 1317.87 | 1.26 | 1316.23 | L11: 211.9 | 42.14 + 84.17 | 19.26 | 0.11 | 3217 → 6410 → 6454 | 4576 |

## Ranked bottleneck table (Step 3 candidate targets)

| Rank | Platform | Bottleneck | Measured evidence | Root cause | Output impact | Expected impact |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **CUDA + Metal** | **Model load latency (`Artifact::load*`)** | 80–96% of load time across all 7 profiles: **5.34 s** (`4.67 s` in `Artifact`) on CUDA LM1B-large, **3.99 s** (`3.68 s`) on Metal LM1B-large, **1.12 s** (`1.04 s`) on Metal LM1B-small vs Python's 0.52 s. | `ncm-io` hashes every bundle **twice** (whole-file + per-tensor) sequentially on 1 CPU thread; Linux x86_64 also lacks `sha2` `asm` (SHA-NI) enabled on `macos/aarch64`. | **Byte-identical** | ~2–4× faster model load on both CUDA and Metal (preserving identical SHA-256 verification). |
| 2 | **Metal (and CUDA host RSS)** | **Model-load peak RSS doubling (2.0× weight size)** | Metal RSS jumps `822 → 1620 MiB` (LM1B-small), `1659 → 3295 MiB` (medium), `3217 → 6410 MiB` (large) and `501 → 992 MiB` (TinyStories), leaving only 0.15–1.2% RSS margin over Python MLX. | `Artifact::load_inference` retains all weight bundles in RAM simultaneously while `from_artifact` allocates full unified-memory MLX arrays; `Lm1bSmallTeacher` / `TinyStoriesTeacher` do not release/stream bundles one at a time. | **Byte-identical** | Cuts Metal RSS on LM1B/TinyStories toward ~1.1–1.3× weight size (~35–45% RSS reduction on LM1B-large, ~2+ GiB saved). |
| 3 | **CUDA** | **LM1B per-level host syncs + unused codec greedy argmax** | LM1B-small b=16 spends `25.9 ms` in per-level sampling (`6.9%`), `17.0 ms` in penalties (`4.5%`), and `32.4 ms` in codec decode (`8.7%`); LM1B-medium/large b=16/8 only lead Python by 4.5–4.8%. | `Lm1bDraw::DeviceFp64` reads back `selected` synchronously on all 11 levels (unlike Metal's deferred single check); `run_impl` also runs `decode_latent_outputs` (computing an unused `[B, 128, 30528]` greedy argmax) even when untraced. | **Byte-identical** | Removes 11 GPU→CPU stalls + unused 62M-element argmax per request; widens CUDA LM1B latency margin. |
| 4 | **Metal** | **LM1B & XSum host FP64 final penalties + unused greedy argmax** | Metal LM1B final penalties take **82.3–84.2 ms** at max batch (**17.4%** of LM1B-small b=16 warm time; **17.1%** at b=1); XSum b=32 penalties take **50.0 ms**. | `LM1B_PENALTY_ROWS = 2` serializes 8 GPU-copy+readback blocks on 2 CPU threads (M1 Ultra has 16 P-cores) and computes an unused `greedy` argmax in `decode_latent_outputs`; inner FP64 penalty loop runs single-threaded per row. | **Byte-identical** | Substantially cuts Metal LM1B warm latency at both b=1 and b=16 while preserving exact FP64 penalty arithmetic and bounded RSS. |
| 5 | **CUDA + Metal** | **Generator transformer layers (`level_00..11`)** | 80–97% of warm inference time at max batch (e.g., XSum b=32 `4.37 s` CUDA / `2.92 s` Metal; LM1B-large b=8 `1.17 s` CUDA / `1.30 s` Metal). | Compute-bound FP32 dense GEMMs/attention across 10–12 hierarchy levels (already using cuDNN/fused attention on CUDA and MLX GPU kernels on Metal). | Potentially output-changing if kernels/graphs change | Requires CUDA graphs, MLX compilation, or new fused kernels; higher risk and requires full quality gates if numerics change. |

## Reproduce

```sh
# Linux / CUDA
PATH="/opt/cuda/bin:$PATH" LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo build --release --locked -p ncm-backend-cuda \
  --features cuda-cudnn,wmt14-experimental,lm1b-experimental --example phase5_stage_profile
LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib:/opt/cuda/lib64" \
  python3 tools/bench/run_phase5_profile.py --platform cuda \
  --binary target/release/examples/phase5_stage_profile --output reports/NEW-profile-cuda

# Apple Silicon / Metal (with MLX_SYS_PREBUILT and RUSTFLAGS set as in docs/USAGE.md)
cargo build --release --locked -p ncm-backend-mlx \
  --features mlx,wmt14-experimental,lm1b-experimental --example phase5_stage_profile
python3 tools/bench/run_phase5_profile.py --platform mlx \
  --binary target/release/examples/phase5_stage_profile --output reports/NEW-profile-mlx
```
