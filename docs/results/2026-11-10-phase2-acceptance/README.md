# Phase 2 final acceptance — CUDA and MLX

**Phase 2, including the reopened performance-remediation extension, passes on
both platforms.** Linux/CUDA work and validation preceded Apple/MLX for every
shared change. The final production candidate is **`083c1e5`**; subsequent changes
are evidence, audit hardening and documentation. Phase 3 is **not started** and
was not implemented under this request. This is not an inference release or full
text-model compatibility claim.

## Gates and preserved contracts

- **78/78 Linux comparisons and 78/78 Apple comparisons pass**, versus both
  previous Rust and qualified Python: median ratio ≤1.05, p95 ≤1.10, comparable
  host RSS/HWM ratio ≤1.05. No thresholds, pins, assets or goldens were changed.
- Final fresh native captures on **each GPU** pass **642 trained traces and 2,500
  code/output decisions**, zero ID disagreements, same two reported near ties.
  Unchanged FP32 elementwise budgets; no exemption for near-tie tensors.
- Full `tools/check.sh --cuda`, then `tools/check.sh --mlx`, pass on real hardware:
  existing operators, components, tiny codec, ties/snapshots, loading/rejection
  checks and relevant adapter regressions, plus formatting, Clippy and docs.
- Cleared-environment real tokenizer probes pass all four profiles and mutation
  rejections. Production remains Python-free. Full trained captures, probes and
  check logs are published as metadata/results here; weights and large traces
  remain outside Git.
- Raw workload reports retain first-call, p95, dispersion, process medians,
  throughput, instrumented diagnostics, RSS/HWM and distinct allocator counters.
  Audits bind native **and Python** control sources to the original matrix.

The detailed mechanisms and rejected experiments remain in
[CUDA remediation](../2026-11-10-phase2-cuda-remediation/README.md),
[shared loading](../2026-11-10-phase2-shared-loading/README.md),
[compact structures](../2026-11-10-phase2-compact-tokenizers/README.md), and
[Apple measurements](../2026-11-10-phase2-mlx-remediation/README.md).

## Final performance and memory highlights

**Linux reconstruction medians, old Rust / candidate / Python (ms):**

| Batch | Old Rust | Candidate Rust | Pinned eager Python |
| --- | ---: | ---: | ---: |
| 1 | 19.6681 | 3.0551 | 4.6880 |
| 2 | 26.7403 | 4.0014 | 5.1000 |
| 4 | 28.9482 | 5.1322 | 5.8610 |

Candidate codec warm load **195.89 ms**, versus Python **293.33 ms**; maximum host
HWM **606.43 MB**, versus Python **1649.81 MB**. Qwen load **65.95 ms**, versus
Python **151.04 ms**; HWM **76.54 MB**, versus **241.64 MB**. These latest full
matrix figures supersede intermediate highlights, without deleting those runs.
Both complete Linux summaries are adjacent; all medians/p95/throughputs and memory
observations remain available, not only the highlighted cells.

**Apple checked load / max host HWM, old Rust → candidate (Python):**

| Scope | Warm load ms | HWM MB, decimal |
| --- | --- | --- |
| Codec | 945.87 → **179.99** (272.21) | 4749.16 → **293.91** (308.69) |
| BERT | 11.84 → **3.47** (17.48) | 30.83 → **15.97** (104.01) |
| GPT-2 | 51.25 → **17.03** (50.91) | 64.47 → **24.99** (228.61) |
| Qwen | 189.29 → **63.65** (144.78) | 633.88 → **69.81** (290.98) |
| TinyStories | 5.63 → **3.80** (30.71) | 11.81 → **10.62** (84.80) |

Apple B1/B2/B4 reconstruction is **24.983 / 27.069 / 29.238 ms**, versus Python
**28.706 / 30.828 / 32.902 ms**, preserving the existing execution advantage.
Full encode/decode/block, tokenizer and allocator metrics are in the linked Apple
summaries. Repeated host-only, array-upload and codec load/drop diagnostics plateau;
MLX active bytes return to zero after each drop. This is evidence for the measured
workloads, not a general assertion that every future workload is leak-free.

## Uncertainty resolved without waivers

The initial Apple matrix's pooled medians/p95 and host-memory ratios passed, but
three process envelopes did not. `raw/initial-mlx-audit.json` preserves that
failure. A fixed **30 fresh triplets for each of GPT-2 and Qwen** now passes all
three execution scopes, versus both old Rust and Python. Every process is kept,
with five warmups/thirty trials; order rotates old/Python/candidate. No executable
identity substitution is used for these changed binaries.

Candidate/old Rust paired-process 95% upper bounds:

| Profile/scope | Median-ratio upper | p95-ratio upper |
| --- | ---: | ---: |
| GPT-2 encode | 1.01709 | 1.02088 |
| GPT-2 decode | 1.00869 | 1.04526 |
| GPT-2 skip-special decode | 1.00908 | 1.01975 |
| Qwen encode | 1.03059 | 1.02493 |
| Qwen decode | 1.02387 | 1.01713 |
| Qwen skip-special decode | 1.03145 | 1.01463 |

Bootstrap: 10,000 paired **whole-process** resamples, seed 20261110 per comparator;
these are descriptive per-cell intervals, not simultaneous statistical guarantees.
macOS has no CPU-affinity claim. Desktop activity is sampled; no unrelated work,
clock, power policy or global tooling was changed. A transient busy audio process
had subsided before controls started. No privileged GPU telemetry is invented.
Additional control-run host-memory maxima also remain favorable: candidate GPT-2
**32.37 MB** versus old **84.38 MB** / Python **245.56 MB**; candidate Qwen
**73.19 MB** versus old **640.52 MB** / Python **302.42 MB**.

Linux retains its fixed tokenizer controls and nine full-codec checked-load
triplets, described in the compact-structures report. All unsuccessful earlier
experiments and uncertainty rounds remain published. The final audits reuse the
unchanged-codec lifecycle evidence on Linux and fresh trained qualifications on
both platforms; reused evidence is not claimed as newly sampled.

## Separate Apple lifecycle

The retrieved report is complete: three fresh processes/runtime, exactly four
ordered internal phases, and identical solved-board output/code digests in all
nine processes. Old/current Rust use byte-identical lifecycle harness source,
compiled separately against the historical/current implementations; the isolated
baseline worktree has only the two added harness files. No historical worktree was
overwritten. Source hashes and state are recorded in `raw/phase2-final-mlx-state.log`.

| Median ms | Previous Rust | Candidate Rust | Adapted Python MLX |
| --- | ---: | ---: | ---: |
| Parent spawn → imports/linker/context ready | 39.839 | 39.683 | 106.101 |
| Internal context initialization | 30.784 | 30.812 | 9.807 |
| Checked load | 942.752 | 178.705 | 277.157 |
| First reconstruction, identity validation included | 61.520 | 61.932 | 63.438 |
| Output/model drop + synchronization | 0.039 | 0.035 | 0.063 |
| Final marker → process exit/context destruction | 6.747 | 6.772 | 17.606 |
| Full process | 1050.719 | 287.297 | 468.007 |

Medians of intervals need not sum. These are warm-filesystem **fresh-process**
observations, not disk-cold or steady-state timings. Startup includes imports and
linking; the child context interval does not. Teardown includes exit work not
captured by model-drop timing. Lifecycle is descriptive, not an invented extra
steady-state non-inferiority gate. The Linux lifecycle is retained in the
shared-loading report and referenced by the final config.

## Reproduction

All evidence needed to regenerate acceptance is committed; no GPU, weights,
ignored reference environment or external reports are needed for the audit:

1. Unpack `linux-capacity-tokenizers-v1.tar.gz`, `linux-capacity-controls-v1.tar.gz`
   and `linux-compact-codec-control-v1.tar.gz` from the compact-structures report's
   `raw/` into `reports/phase2-performance-linux-<stem>/` (omit `linux-` from stem).
2. Unpack Apple remediation `raw/threeway-v1.tar.gz` into
   `reports/phase2-performance-mlx-remediated-v1/` and this report's
   `raw/mlx-controls-v1.tar.gz` into `reports/phase2-performance-mlx-controls-v1/`.
3. Copy `raw/{cuda,mlx}-qualification.json` to
   `reports/phase2-codec/{cuda,mlx}-phase2-final-qualification.json`.
   Copy `raw/phase2-mlx-lifecycle-v1.json` into `reports/`; copy shared-loading
   `raw/linux-lifecycle.json` to `reports/phase2-linux-shared-lifecycle-v1.json`.
4. Run the two configs with new output paths:

```sh
python3 tools/bench/audit_phase2_performance.py \
  docs/results/2026-11-10-phase2-acceptance/raw/phase2-final-linux-config.json reports/linux-final.json
python3 tools/bench/audit_phase2_performance.py \
  docs/results/2026-11-10-phase2-acceptance/raw/phase2-final-mlx-config.json reports/mlx-final.json
```

A clean staged checkout with no ignored artifacts exactly regenerated **both
78-cell audits and all four old/candidate matrix summaries**, verified evidence
hashes, and passed all **24 report-tool tests**. Control executable hashes were
also checked against their original matrices; previous raw evidence is unchanged.
The 21 offline reference-tool tests and final default publication checks also pass;
their logs are included.

`evidence.json` records file digests. Final checks were run Linux first, then Mac:
`NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`;
Mac uses the pinned `.reference/mlx-native` prefix/rpath and two build jobs for
`./tools/check.sh --mlx`. Fresh native trace capture on CUDA, then Metal, was
qualified by `tools/reference/qualify_codec.py` against the unchanged trained CUDA
oracle. No tolerance or golden regeneration was performed.

## Boundaries

These results cover the FP32 Sudoku codec and four tokenizers—not generation,
full text-model inference, official Sudoku-solving quality, or a release. The
lossy codec's historical 437/445 input recovery is not 100% reconstruction or
solving accuracy. Python on Apple is an explicitly qualified adaptation, not
unmodified upstream Apple execution. Source/canonical tokenizer formats differ;
native inference omits reference training buffers. Warm filesystem cache and
workstation variability are disclosed.

CUDA default-pool counters are not equivalent to Torch allocator/process VRAM.
The 128 MiB pool release threshold retains reusable reservation (up to 32 MiB more
in the measured workload), not additional live model data or a hard allocation
limit; it affects the process's shared per-device default pool. MLX active/cache
and process RSS are distinct unified-memory metrics. No universal device-memory
advantage is inferred. Large licensed trace/tokenizer release archives remain
separately pending publication authorization; no weights/data were added to Git.
