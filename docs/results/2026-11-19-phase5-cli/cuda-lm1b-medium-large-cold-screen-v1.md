# Phase 5 CUDA CLI cold-request screens: LM1B medium and large

2026-11-19. These exploratory **Rust baseline versus candidate** screens are not release performance
gates or Rust-versus-Python comparisons. Each measures one seeded single-row request in a fresh
process, including startup, artifact/rotary loading, generation, readback and JSON output. Separate
warmup processes warm the system, not a persistent model. Device-only spans, native CUDA
allocation/reservation peaks, per-level times, Python comparison, throughput sweeps and full quality
results are absent. The active desktop and these omissions leave the
[reporting protocol](../../REPORTING.md) unmet for release.

The pre-change `702168c` baseline uses `sample-lm1b` with seed 42 and an explicit rotary path
(binary SHA-256 `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`); the candidate
uses unified `sample --seed 42 --reference-rope` (binary SHA-256
`cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`). Both use `cuda-cudnn`,
the pinned rotary table SHA-256 `9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`
and input `[[101]]` SHA-256 `a283936d3db1e156d26a0ace722f476f1895782f23729587a45c4f838168e7dc`.
All 210 warmup and measured processes in each screen produced identical stdout. This fixed-request
identity is not task-level quality; the default rotary lookup is not measured.

| Profile | Qualified manifest SHA-256 | Common stdout SHA-256 |
| --- | --- | --- |
| Medium | `2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c` | `d9c1c40663e66cf472846e3781ca4995bab998b5bc86ec0c675fe0b29c506488` |
| Large | `86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a` | `bbdc24f6d0bd2150c41af25fb4a37835e9a191fcce5fe181461863421b9bfbf9` |

Each screen has three groups of five warmup and 30 measured paired processes per condition,
alternating baseline/candidate order. Python `perf_counter_ns` spans GNU time launching the CLI
through process exit and captured output, including wrapper and file I/O; GNU time `%M` is host RSS,
not GPU memory. The table pools 90 measured processes per condition. The bootstrap resamples the 90
within-round candidate-minus-baseline differences 10,000 times with Python RNG seed 42; it is
descriptive and does not prove independence or eliminate drift.

| Profile | Condition | Median ms | p95 ms | MAD ms | Maximum ms | Median peak RSS MiB | Maximum peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Medium | Baseline | 2727.627 | 2772.705 | 10.418 | 2787.723 | 2048.752 | 2049.891 |
| Medium | Candidate | 2727.322 | 2754.315 | 7.013 | 2763.533 | 2044.080 | 2047.508 |
| Large | Baseline | 4851.750 | 4877.999 | 10.118 | 4899.491 | 3605.498 | 3608.445 |
| Large | Candidate | 4856.100 | 4886.873 | 9.241 | 4913.631 | 3603.758 | 3605.414 |

Medium's pooled candidate/baseline median ratio was 0.99989; median paired difference +1.373 ms,
with descriptive bootstrap interval [-5.182, 5.775] ms. Group medians were baseline
2725.332/2729.930/2729.312 and candidate 2724.382/2730.296/2729.092 ms. Large's ratio was
1.00090; median paired difference +1.335 ms, interval [-2.226, 5.501] ms. Group medians were
baseline 4856.736/4849.645/4854.903 and candidate 4854.829/4856.159/4854.894 ms. Neither screen
establishes a meaningful improvement or non-regression margin.

The host was an RTX 3060 Ti, driver 620.70.06, CUDA 13.4 and reference cuDNN. The runner hashes
itself, binaries, manifest, input, rotary table, GNU time, `nvidia-smi` and selected CUDA/cuDNN
libraries before and after; this is not whole-process mapped-library coverage. Medium's 574
system-wide one-second GPU samples ranged 45.00–52.69 °C, 210–1965 MHz SM clock, 22.66–80.52 W,
0–100% utilization and 1,658–3,613 MiB total GPU memory. Large's 1,019 samples ranged
42.62–52.84 °C, 210–1965 MHz, 22.64–96.26 W, 0–100% and 1,658–5,066 MiB. They include other
processes, can miss short peaks, and are not native CLI device peaks. No GPU reservation or absence
of competing work was established.

## Reproduce

Runner SHA-256 is `840b7fd7c1318a68f13a7faf98407791085858efaeaa7b469f71c3ece19b52e6`. Ignored
result SHA-256s: medium `cef345bf90c3aaf498c2973965a8dff32c524ad031f7354984e5719e13160f7b`,
large `0a0907ae6d7eeac16b5dfccdb8872dda2c89f992e18da03785e33a4c85cb783c`. Sampled GPU CSV
SHA-256s: medium `906bf02c488f512f8d3d01cf19863a1bfb6b7875323d1c1fd6ebca6d7c4ff263`, large
`4769ee2ea9d1ba9c552efd767792df3c2d1c55ce6e96d479c910584942eb8744`. The following runs both
screens in sequence with fresh output paths; raw outputs and telemetry stay ignored.

```sh
for size in medium large; do
  if [ "$size" = medium ]; then
    manifest=2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c
    expected=d9c1c40663e66cf472846e3781ca4995bab998b5bc86ec0c675fe0b29c506488
  else
    manifest=86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a
    expected=bbdc24f6d0bd2150c41af25fb4a37835e9a191fcce5fe181461863421b9bfbf9
  fi
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
    python3 tools/bench/phase5_cli_cold_screen.py --profile lm1b \
    --baseline reports/phase5-cli-equivalence/baseline-ncm \
    --baseline-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
    --candidate target/phase5-candidate/release/ncm \
    --candidate-sha256 cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67 \
    --artifact "artifacts/canonical/lm1b-$size-inference-v1" --manifest-sha256 "$manifest" \
    --input reports/phase5-cli-equivalence-runner-v3/lm1b-input.json \
    --input-sha256 a283936d3db1e156d26a0ace722f476f1895782f23729587a45c4f838168e7dc \
    --expected-output-sha256 "$expected" \
    --rope reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
    --rope-sha256 9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc \
    --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
    --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
    --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
    --output "reports/NEW-lm1b-$size-cold-screen" || exit 1
done
```

Recompute each paired interval from non-warmup records grouped by `(group, ordinal)`: subtract
baseline seconds from candidate, resample with `random.Random(42).choices` 10,000 times and sort
the medians; positions 249 and 9749 give the displayed endpoints. The [small screen](cuda-lm1b-cold-screen-v1.md)
uses the same runner. These screens measure batch 1 only and do not validate the 8 GiB card's
capacity at larger batches.
