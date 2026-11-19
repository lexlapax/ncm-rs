# Phase 5 CUDA CLI cold-request screen: Sudoku easy

2026-11-19. This is an exploratory **Rust baseline vs candidate**, not a release performance gate
or a Rust-vs-Python comparison. It covers one two-row Sudoku-easy request on one desktop GPU; it
measures fresh process startup, artifact load, generation, JSON output and readback together. Each
request is a new process: the five preceding warmups per group warm the system, **not** a persistent
model. There are no device-only spans, native CUDA allocation/reservation peaks, per-level times,
Python comparison, throughput sweep or full quality result. These omissions and desktop activity
mean the [reporting protocol](../../REPORTING.md) remains unmet for release.

## Setup and output identity

The pre-change baseline is `702168c`'s `generate` binary (SHA-256
`f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`); the candidate is the
`16bb532` CLI build with `sample` (SHA-256
`cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`). Later commits through
`17ee25f` did not change production CLI code. Both used the qualified Sudoku-easy manifest
`ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832` and the two-row input
`7dbb5c739c25e6f8a1ede490278261d4d800ec8529a14c038b08e11a4278c41c`. All 210 warmup and
measured processes produced identical stdout SHA-256
`bd2a86aa8d3c626178bfd1d07bd2bbbdfee2a98643298c02c189e59bb2a06212`.
This is fixed-request identity, not task-level quality.

The tracked runner executes three groups, each containing five warmup and 30 measured paired
requests per condition; baseline/candidate order alternates. Python `perf_counter_ns` spans
`/usr/bin/time` launching the CLI through process exit and captured stdout, including wrapper and
file I/O. GNU time `%M` reports each child process's maximum RSS; it is not GPU memory. Median,
nearest-rank pooled p95, median absolute deviation (MAD) and maxima below summarize 90 measured
fresh processes per condition. The paired bootstrap resamples the 90 within-round candidate-minus-
baseline elapsed differences 10,000 times with Python RNG seed 42; it is descriptive and does not
prove independence or remove thermal/background drift.

| Condition | Median ms | p95 ms | MAD ms | Maximum ms | Median peak RSS MiB | Maximum peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 555.115 | 617.828 | 13.013 | 631.245 | 587.965 | 590.102 |
| Candidate | 554.145 | 589.608 | 13.512 | 615.830 | 587.773 | 590.766 |

The candidate/baseline pooled median ratio was 0.99825; median paired difference was -1.257 ms,
with descriptive bootstrap interval [-4.178, 2.914] ms. This screen does not establish a
meaningful latency improvement or a non-regression margin. Group median ms were baseline
559.253/560.628/536.243 and candidate 561.067/562.560/535.824.

The host was an RTX 3060 Ti, NVIDIA driver 620.70.06, CUDA 13.4 and the reference cuDNN runtime.
The runner fingerprints itself, binaries, artifact manifest, input, GNU time, `nvidia-smi` and the
specified CUDA/cuDNN libraries before and after; this is not whole-process mapped-library coverage.
The desktop was active; no GPU reservation or absence of competing work was established. The 117
system-wide one-second `nvidia-smi` samples ranged 40.47–46.84 °C, 210–1710 MHz SM clock,
22.60–61.24 W, 0–45% utilization and 1,788–2,151 MiB total GPU memory. These samples can miss
short peaks and include other processes; they are **not** the CLI's native device peak.

Three earlier screens (`reports/phase5-cli-cold-sudoku-easy-v1/` through `v3/`) used a timed
Python process wait that polls in coarse intervals and biases the elapsed measurements. Their
latencies are invalid and are not compared; their raw outputs remain retained locally. The final
screen uses a blocking wait with an alarm timeout and isolated process-group cleanup.

## Reproduce

The runner used for this screen had SHA-256 `a7991823d74165395b0a95219e9a876be06ed42e8c8c75f528ed4acbeb45c0d3`.
It was later extended for a separate [TinyStories screen](cuda-tinystories-cold-screen-v1.md);
that updated runner was not used for the Sudoku measurements below.
The ignored raw result SHA-256 is `279698ef160131ecc3ca793ecb32071a70f8fc2d8233444b24e3d90b28d4888a`;
the sampled GPU CSV SHA-256 is `009da93a40b4ed50e626ab9a220bf6e7f15510c9d5d0772ac3fc1497a8e812d8`.
The fixed input is evaluation-derived and is not committed. Use a fresh output directory; raw
outputs, timings and telemetry stay ignored and may contain evaluation-derived data.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
  LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
  python3 tools/bench/phase5_cli_cold_screen.py \
  --baseline reports/phase5-cli-equivalence/baseline-ncm \
  --baseline-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
  --candidate target/phase5-candidate/release/ncm \
  --candidate-sha256 cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67 \
  --artifact artifacts/canonical/sudoku-easy-qualified \
  --manifest-sha256 ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832 \
  --input artifacts/canonical/sudoku-data-phase3/easy-two-inputs.json \
  --input-sha256 7dbb5c739c25e6f8a1ede490278261d4d800ec8529a14c038b08e11a4278c41c \
  --expected-output-sha256 bd2a86aa8d3c626178bfd1d07bd2bbbdfee2a98643298c02c189e59bb2a06212 \
  --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
  --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
  --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
  --output reports/NEW-cli-cold-screen
```

The paired interval can be recomputed from `result.json` by grouping non-warmup records by
`(group, ordinal)`, subtracting baseline elapsed seconds from candidate, and bootstrapping the
median of those 90 differences with `random.Random(42).choices` 10,000 times; sorted positions
249 and 9749 give the displayed endpoints. The runner's tests cover summary counts, positive RSS,
output mismatches, process failures, timeout cleanup and an existing dangling output symlink.
