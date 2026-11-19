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

The raw directory is still only on the Mac. It will be copied to Linux when the Mac is available again.

Reproduce on the Mac from the repository root after building the examples with
`--features mlx,lm1b-experimental`:

```sh
.reference/mlx-venv/bin/python tools/bench/run_lm1b_small_mlx_timing.py \
  --rotary PATH/TO/reference-rope-tables.safetensors --output reports/NEW-DIRECTORY
```
