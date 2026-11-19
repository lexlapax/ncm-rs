# Phase 5 CUDA CLI cold-request screens: XSum and WMT14

2026-11-19. These exploratory **Rust baseline versus candidate** screens are not release performance
gates or Rust-versus-Python comparisons. Each measures one single-row top-1, CFG-0 request in a
fresh process, including startup, artifact loading, generation, readback and JSON output. The
separate warmup processes warm the system, not a persistent model. Device-only spans, native CUDA
allocation/reservation peaks, per-level times, Python comparison, throughput sweeps and full quality
results are absent. The active desktop and these omissions leave the
[reporting protocol](../../REPORTING.md) unmet for release.

The pre-change `702168c` baseline uses `sample-xsum` or `sample-wmt14` (binary SHA-256
`f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`); the candidate uses unified
`sample` (binary SHA-256 `cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`).
XSum runs `cuda-cudnn`, WMT14 `cuda`. Qualified artifact, input and common stdout identities:

| Profile | Manifest SHA-256 | Input SHA-256 | Stdout SHA-256 |
| --- | --- | --- | --- |
| XSum | `8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4` | `74703e40c8fd26fff5a6d919c6d40550a39a33b94183455739093947b7ef2489` | `87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca` |
| WMT14 | `01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81` | `8b0fd6d29cacfd8acf1e033a3664aec21c2a28976af9f9fc1ad77d62b1c8a825` | `17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a` |

All 210 warmup and measured processes in each screen produced the respective identical stdout.
These fixed-request identities are not task-level quality. The input rows are evaluation-derived
and are not committed.

Each screen has three groups of five warmup and 30 measured paired processes per condition,
alternating baseline/candidate order. Python `perf_counter_ns` spans GNU time launching the CLI
through exit and output capture, including wrapper and file I/O; GNU time `%M` is host RSS, not GPU
memory. The table pools 90 measured processes per condition. The bootstrap resamples the 90
within-round candidate-minus-baseline differences 10,000 times with Python RNG seed 42; it is
descriptive and does not prove independence or remove drift.

| Profile | Condition | Median ms | p95 ms | MAD ms | Maximum ms | Median peak RSS MiB | Maximum peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XSum | Baseline | 1285.126 | 1302.423 | 6.412 | 1316.417 | 988.980 | 989.703 |
| XSum | Candidate | 1283.963 | 1299.876 | 4.976 | 1310.646 | 989.160 | 989.758 |
| WMT14 | Baseline | 1837.986 | 1866.316 | 7.372 | 1901.486 | 1463.812 | 1464.113 |
| WMT14 | Candidate | 1834.103 | 1871.558 | 8.535 | 1889.067 | 1463.891 | 1464.301 |

XSum's pooled candidate/baseline median ratio was 0.99910; median paired difference was -2.056 ms,
with descriptive bootstrap interval [-5.139, 2.066] ms. Group medians were baseline
1284.565/1284.744/1288.071 and candidate 1284.111/1283.963/1284.141 ms. WMT14's ratio was
0.99789; median paired difference was -2.709 ms, interval [-9.802, 2.803] ms. Group medians were
baseline 1838.375/1839.330/1834.135 and candidate 1834.635/1838.978/1830.208 ms. Neither screen
establishes a meaningful improvement or non-regression margin.

The host was an RTX 3060 Ti, driver 620.70.06, CUDA 13.4 and reference cuDNN. The runner hashes
itself, binaries, manifest, input, GNU time, `nvidia-smi` and selected CUDA/cuDNN libraries before
and after; this is not whole-process mapped-library coverage. XSum's 270 system-wide one-second GPU
samples ranged 43.34–56.44 °C, 210–1980 MHz SM clock, 22.47–89.74 W, 0–100% utilization and
1,740–2,584 MiB total GPU memory. WMT14's 387 samples ranged 45.16–51.34 °C, 210–1980 MHz,
24.00–73.51 W, 0–77% and 1,741–3,001 MiB. They include other processes, can miss short peaks, and
are not native CLI device peaks. No GPU reservation or absence of competing work was established.

## Reproduce

The runner used here had SHA-256 `782ff20af6e6a01db03c7131b89664dc0333eafb5c97149abb949bb81ffaca3e`;
it was later extended for a separate [LM1B-small screen](cuda-lm1b-cold-screen-v1.md). Ignored
result SHA-256s: XSum `c0d2e9f27a87e0de85c098c866d63329f09be307f42cd93971b315ad5a3ed3d1`,
WMT14 `45d58ead0d1b988bbb4572f105d4cd186acb4889c44f09fbaf1ded2d19b0f713`. Sampled GPU CSV
SHA-256s: XSum `4b756826ec3d663688a56d77a77db933c787c669d61fd2c9711a1e6d06201a48`, WMT14
`f608605bc1565530364941f862404c430e7f9e0e185aa1facddcb0a19a988c37`. The command below reproduces
the two runs in sequence with fresh output paths. Raw outputs and telemetry stay ignored and may
contain evaluation-derived data.

```sh
for profile in xsum wmt14; do
  if [ "$profile" = xsum ]; then
    artifact=artifacts/canonical/xsum-inference-rope-v2-final
    manifest=8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4
    input=74703e40c8fd26fff5a6d919c6d40550a39a33b94183455739093947b7ef2489
    expected=87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca
  else
    artifact=artifacts/canonical/wmt14-inference-v1-final
    manifest=01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81
    input=8b0fd6d29cacfd8acf1e033a3664aec21c2a28976af9f9fc1ad77d62b1c8a825
    expected=17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a
  fi
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
    python3 tools/bench/phase5_cli_cold_screen.py --profile "$profile" \
    --baseline reports/phase5-cli-equivalence/baseline-ncm \
    --baseline-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
    --candidate target/phase5-candidate/release/ncm \
    --candidate-sha256 cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67 \
    --artifact "$artifact" --manifest-sha256 "$manifest" \
    --input "reports/phase5-cli-equivalence/$profile.json" --input-sha256 "$input" \
    --expected-output-sha256 "$expected" \
    --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
    --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
    --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
    --output "reports/NEW-$profile-cold-screen" || exit 1
done
```

Recompute each paired interval from non-warmup records grouped by `(group, ordinal)`: subtract
baseline seconds from candidate, resample with `random.Random(42).choices` 10,000 times and sort
the medians; positions 249 and 9749 give the displayed endpoints. Seven offline tests cover summary
counts, routing, output identity, failures, timeout cleanup and output symlinks. The earlier
[Sudoku](cuda-sudoku-cold-screen-v1.md) and [TinyStories](cuda-tinystories-cold-screen-v1.md)
screens retain their own runner identities.
