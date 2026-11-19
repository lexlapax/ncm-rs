# Phase 5 CUDA CLI cold-request screen: LM1B small

2026-11-19. This exploratory **Rust baseline versus candidate** screen is not a release performance
gate or a Rust-versus-Python comparison. It measures one single-row seeded LM1B-small request in a
fresh process, including startup, artifact/rotary loading, generation, readback and JSON output.
Separate warmup processes warm the system, not a persistent model. Device-only spans, native CUDA
allocation/reservation peaks, per-level times, Python comparison, throughput sweeps and full quality
results are absent. The active desktop and these omissions leave the
[reporting protocol](../../REPORTING.md) unmet for release.

The pre-change `702168c` baseline uses `sample-lm1b` with seed 42 and an explicit rotary path
(binary SHA-256 `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`); the candidate
uses unified `sample --seed 42 --reference-rope` (binary SHA-256
`cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`). Both run `cuda-cudnn` with
qualified manifest `58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76`, rotary
SHA-256 `9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc` and input `[[101]]`,
SHA-256 `a283936d3db1e156d26a0ace722f476f1895782f23729587a45c4f838168e7dc`. All 210 warmup and
measured processes produced identical stdout SHA-256
`520ae6b5372b2013bda8235dfec40c85265716014ac4a9ba58fec03419be5d71`. This fixed-request identity
is not task-level quality; the default rotary lookup is not measured here.

Three groups each use five warmup and 30 measured paired processes per condition, alternating
baseline/candidate order. Python `perf_counter_ns` spans GNU time launching the CLI through process
exit and captured output, including wrapper and file I/O; GNU time `%M` is maximum host RSS, not GPU
memory. The table pools 90 measured processes per condition. The bootstrap resamples the 90
within-round candidate-minus-baseline differences 10,000 times with Python RNG seed 42; it is
descriptive and does not prove independence or eliminate drift.

| Condition | Median ms | p95 ms | MAD ms | Maximum ms | Median peak RSS MiB | Maximum peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1578.016 | 1613.419 | 6.014 | 1645.711 | 1209.566 | 1212.840 |
| Candidate | 1578.826 | 1612.661 | 7.253 | 1768.431 | 1206.109 | 1208.336 |

The pooled median ratio was 1.00051; median paired difference was +0.602 ms with descriptive
bootstrap interval [-1.716, 3.952] ms. Group medians were baseline 1578.963/1578.016/1577.346 ms
and candidate 1578.166/1578.476/1579.687 ms. This does not establish a meaningful improvement or
non-regression margin.

The host was an RTX 3060 Ti, driver 620.70.06, CUDA 13.4 and reference cuDNN. The runner hashes
itself, binaries, artifact manifest, input, rotary table, GNU time, `nvidia-smi` and selected
CUDA/cuDNN libraries before and after; this is not whole-process mapped-library coverage. The 333
system-wide one-second GPU samples ranged 44.34–48.41 °C, 210–1710 MHz SM clock, 22.37–56.80 W,
0–64% utilization and 1,740–2,748 MiB total GPU memory. They include other processes, can miss
short peaks and do not measure CLI device peaks. No GPU reservation or absence of competing work
was established.

## Reproduce

Runner SHA-256 is `840b7fd7c1318a68f13a7faf98407791085858efaeaa7b469f71c3ece19b52e6`; ignored
result SHA-256 is `08c5013f84c469ad34e3ea3f99f1a2c32deed51993f095cc7ba1ebdaecd77450` and
sampled GPU CSV SHA-256 is `1070ff9a3cf239cc67a0eabb77677f7e2ba6b679c00dcf9cbb099721ed9c7d8a`.
Use a fresh output directory; raw captures stay ignored.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
  LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
  python3 tools/bench/phase5_cli_cold_screen.py --profile lm1b \
  --baseline reports/phase5-cli-equivalence/baseline-ncm \
  --baseline-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
  --candidate target/phase5-candidate/release/ncm \
  --candidate-sha256 cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67 \
  --artifact artifacts/canonical/lm1b-small-inference-v1 \
  --manifest-sha256 58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76 \
  --input reports/phase5-cli-equivalence-runner-v3/lm1b-input.json \
  --input-sha256 a283936d3db1e156d26a0ace722f476f1895782f23729587a45c4f838168e7dc \
  --expected-output-sha256 520ae6b5372b2013bda8235dfec40c85265716014ac4a9ba58fec03419be5d71 \
  --rope reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
  --rope-sha256 9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc \
  --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
  --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
  --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
  --output reports/NEW-lm1b-cold-screen
```

The paired interval can be recomputed by grouping non-warmup records by `(group, ordinal)`,
subtracting baseline seconds from candidate, and bootstrapping the median with
`random.Random(42).choices` 10,000 times; sorted positions 249 and 9749 are the endpoints. Nine
runner tests cover counts, profile/seed/rotary routing, missing rotary arguments, output identity,
failures, timeout cleanup and output symlinks. Prior [Sudoku](cuda-sudoku-cold-screen-v1.md),
[TinyStories](cuda-tinystories-cold-screen-v1.md) and [XSum/WMT14](cuda-xsum-wmt14-cold-screen-v1.md)
screens retain their respective runner identities.
