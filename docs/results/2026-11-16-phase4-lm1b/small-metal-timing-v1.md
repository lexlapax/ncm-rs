# LM1B-small Metal latency and memory

Rust Metal was compared with the Python MLX adaptation on the Mac Studio (M1 Ultra, 64 GiB) at
commit `0a3ce34`. The runner is `tools/bench/run_lm1b_small_mlx_timing.py`. For each of batches 1, 4
and 16 it runs three fresh processes per runtime in alternating order. Each process does 5 warmups
and 30 measured requests, so each cell has 90 samples per runtime. Peak process RSS comes from
`/usr/bin/time -l`. The measured scope is generation, decoding penalties and code/ID readback;
model load is reported separately.

- **Rust:** the CLI's default Metal path, `sample_seeded_device_fp32` (device FP32 inverse CDF, host
  FP64 penalties).
- **Python:** the qualified development reference's forward pass and FP64 penalties
  (`tools/reference/lm1b_small_mlx_reference.py`), with on-device `mx.random.categorical` sampling,
  as the TinyStories Python MLX baseline does (`tools/reference/lm1b_small_mlx_timing_probe.py`).
  The reference's exact controlled sampler is a pure-Python loop over every 8,192-way distribution
  and would not be a fair speed baseline. Random streams differ, so outputs are not compared here.
  Quality is covered by the [80-seed Metal result](cohort-campaign-v1.md#prospective-metal-result).

| Batch | Python median / p95 (s) | Rust median / p95 (s) | Python RSS (MiB) | Rust RSS (MiB) | Result |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 0.1171 / 0.1176 | 0.1015 / 0.1019 | 1,681 | 1,658 | pass |
| 4 | 0.2318 / 0.2328 | 0.1852 / 0.1863 | 1,681 | 1,705 | **RSS fails (+1.4%)** |
| 16 | 0.6606 / 0.6694 | 0.4311 / 0.4332 | 1,681 | 1,882 | **RSS fails (+12.0%)** |

Rust is 13%, 20% and 35% faster at the median and also lower at p95 in every cell. Per-process RSS
was stable to 1 MiB within each cell. MLX allocator peaks (Rust/Python, MiB) were 927/943, 1,296/1,363
and 2,174/2,691. Median model load was 1.14 s for Rust and 0.52 s for Python. Load is reported, not gated.

Python's RSS is the same 1,681 MiB at every batch, which suggests its peak is set during model
loading. Rust's RSS grows with batch by about the size of the decoder logits that the host FP64
penalties read: +47 MiB against 60 MiB of logits at batch 4, and +224 MiB against 238 MiB at
batch 16. This is a hypothesis from the numbers, not a verified cause. The two memory failures stay
visible. The one-minute load average was about 1.1–1.4 before the run and 1.38 after it.

A short screen preceded this run and is retained. It found the previous host FP64 Metal sampler 34–47%
slower than Python. The sampler change and its checks are recorded in the context map at `0a3ce34`.

| Item | SHA-256 |
| --- | --- |
| `result.json` (Mac `reports/phase4-lm1b-mlx-timing-v1/`) | `b574c7c663565a748f09e0414a229afe436801cb36e65ed5ad987b67373b619c` |
| `manifest.json` | `456ad006ab41a7a457fcdb1347025f11d687cad39391c01ab33ea71a35c8fd9d` |

The raw directory has since been copied to Linux as ignored `reports/phase4-lm1b-mlx-timing-v1/` (same `result.json` hash).

## After the memory fix (all checks pass)

Commit `57e5ee6` changed how host FP64 penalties read the decoder logits: they now visit two rows at a
time (`visit_f32_rows`, `LM1B_PENALTY_ROWS = 2`), and on MLX each block is its own exact GPU copy, so the
full logits buffer is never mapped to the CPU. Outputs are unchanged. The CLI at `0a3ce34` and at
`30349ae` produced byte-identical JSON for development seed 11 at batches 1, 4 and 16, with both the
default sampler and `--host-fp64` (ignored `reports/phase4-lm1b-rss-fix-identity-v1/`). The five real-Metal
sampler tests, including the new row-visit test, pass, as does `tools/check.sh --mlx`.

The same timing protocol, rerun at `30349ae` on 2026-11-19:

| Batch | Python median / p95 (s) | Rust median / p95 (s) | Python RSS (MiB) | Rust RSS (MiB) | Result |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 0.1172 / 0.1176 | 0.1020 / 0.1024 | 1,681 | 1,674 | pass |
| 4 | 0.2322 / 0.2338 | 0.1977 / 0.2006 | 1,682 | 1,675 | pass |
| 16 | 0.6621 / 0.6646 | 0.4743 / 0.4769 | 1,682 | 1,672 | pass |

Rust's RSS no longer grows with batch, which confirms the hypothesis above: the growth was the
CPU-mapped decoder logits. The RSS margin is small (0.4–0.6%), because both processes' peaks are now
set by model loading. Rust is 13%, 15% and 28% faster at the median. Batch 4 and 16 latency rose
slightly from the first run (0.185 to 0.198 s and 0.431 to 0.474 s): reading the logits in blocks
costs some speed in exchange for the memory saving. `result.json` SHA-256:
`2b976e54b7f069034c1808c1ad098e2da0cc686dbc0acfef0afb1dd64a061926` (ignored `reports/phase4-lm1b-small-mlx-timing-v2/`).

Reproduce on the Mac from the repository root after building the examples with
`--features mlx,lm1b-experimental`:

```sh
.reference/mlx-venv/bin/python tools/bench/run_lm1b_small_mlx_timing.py \
  --size small --rotary PATH/TO/reference-rope-tables.safetensors --output reports/NEW-DIRECTORY
```
