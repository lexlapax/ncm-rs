# Phase 5 CUDA CLI cold-request screen: TinyStories

2026-11-19. This exploratory **Rust baseline versus candidate** screen is not a release performance
gate or a Rust-versus-Python comparison. It measures one single-row TinyStories request in a fresh
process, including startup, artifact loading, generation, readback and JSON output. The separate
warmup processes warm the system, not a persistent model. There are no device-only spans, native
CUDA allocation/reservation peaks, per-level times, Python comparison, throughput sweep or full
quality results. The active desktop and these omissions leave the
[reporting protocol](../../REPORTING.md) unmet for release.

The pre-change `702168c` baseline uses `sample-tinystories` with seed 42 (binary SHA-256
`f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`); the candidate uses unified
`sample --seed 42` (binary SHA-256 `cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`).
Both use the qualified TinyStories manifest
`0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1` and input `[[0]]`, SHA-256
`3aa8ac761c5a23906d30fb86ca2db09c9b391142a956cdb4263570adb94bfbe6`. All 210 warmup and
measured stdout streams matched SHA-256
`7c828e7bcf538bd65a60762429e4df318c6d71a52e3d8cb20210bfaf14eb6837`. This fixed-request identity
is not task-level quality.

Three groups each use five warmup and 30 measured paired processes per condition, alternating
baseline/candidate order. Python `perf_counter_ns` spans GNU time launching the CLI through process
exit and output capture, including wrapper and file I/O; GNU time `%M` is maximum host RSS, not GPU
memory. The table pools 90 measured processes per condition. The bootstrap resamples the 90
within-round candidate-minus-baseline differences 10,000 times with Python RNG seed 42; it is
descriptive and does not prove independence or eliminate drift.

| Condition | Median ms | p95 ms | MAD ms | Maximum ms | Median peak RSS MiB | Maximum peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1103.932 | 1122.229 | 5.454 | 1141.374 | 908.039 | 910.973 |
| Candidate | 1103.759 | 1118.419 | 6.183 | 1129.870 | 907.695 | 910.641 |

The pooled median ratio was 0.99984; median paired difference was +0.618 ms, with descriptive
bootstrap interval [-1.463, 3.016] ms. Group medians were baseline 1102.759/1103.993/1103.683 ms
and candidate 1101.901/1103.888/1105.376 ms. This does not establish a meaningful improvement or
non-regression margin.

The host was an RTX 3060 Ti, driver 620.70.06, CUDA 13.4 and reference cuDNN. The runner hashes
itself, binaries, artifact manifest, input, GNU time, `nvidia-smi` and selected CUDA/cuDNN libraries
before and after; this is not whole-process mapped-library coverage. The 233 system-wide one-second
GPU samples ranged 43.34–47.69 °C, 210–1710 MHz SM clock, 22.51–56.23 W, 0–55% utilization and
1,725–2,410 MiB total GPU memory. They include other processes, can miss short peaks, and do not
measure the CLI's device peak. No GPU reservation or absence of competing work was established.

## Reproduce

The runner used here had SHA-256 `f0ce2842f26ce4625ba532abd2fd83b0960713cf5b8ef4c07daab348bb12ba06`;
it was later extended for separate [XSum/WMT14 screens](cuda-xsum-wmt14-cold-screen-v1.md).
The ignored result SHA-256 is `2db9e7df31f7001351c251d9526432fd4eb63ff936c1bf70d30f7d5156b3c6b4`
and GPU CSV SHA-256 is `8ce854bd5d867805a5b30de999d91040c16c2c9b824735527b2007d2159b0326`.
Use a fresh output directory; raw files stay ignored.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
  LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
  python3 tools/bench/phase5_cli_cold_screen.py --profile tinystories \
  --baseline reports/phase5-cli-equivalence/baseline-ncm \
  --baseline-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
  --candidate target/phase5-candidate/release/ncm \
  --candidate-sha256 cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67 \
  --artifact artifacts/canonical/tinystories-phase4-v1 \
  --manifest-sha256 0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1 \
  --input reports/phase5-cli-equivalence/tiny.json \
  --input-sha256 3aa8ac761c5a23906d30fb86ca2db09c9b391142a956cdb4263570adb94bfbe6 \
  --expected-output-sha256 7c828e7bcf538bd65a60762429e4df318c6d71a52e3d8cb20210bfaf14eb6837 \
  --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
  --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
  --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
  --output reports/NEW-tinystories-cold-screen
```

The bootstrap can be recomputed by grouping non-warmup records by `(group, ordinal)`, subtracting
baseline seconds from candidate, then taking the median of `random.Random(42).choices` resamples
10,000 times; sorted positions 249 and 9749 give the displayed endpoints. Six offline runner tests
cover summary counts, routing/seed, output identity, failures, timeout cleanup and an output
symlink. The separate [Sudoku screen](cuda-sudoku-cold-screen-v1.md) retains its own runner identity.
