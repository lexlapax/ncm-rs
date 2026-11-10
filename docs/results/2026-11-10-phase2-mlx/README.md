# Phase 2 Apple/MLX gate: codec and tokenizer measurements

**Apple/MLX Phase 2 acceptance is complete.** This closes the measured Apple gate
for the shared FP32 Sudoku codec and pinned tokenizer assets after the completed
Linux/CUDA gate. It is not an NCM generation benchmark, full text-model support,
a model-quality result, or a pure Rust-versus-Python speed claim.

## Evidence, identity and protocol

- [MLX codec/tokenizer parity](../2026-11-10-phase2-mlx-codec/README.md) passed all
  18 unchanged operator cases, 140 component comparisons, 236 tiny traces, all
  642 trained CUDA-oracle traces and 2,500 exact code/output decisions. It retains
  the initial failed long-dot capture, correction, two mixed-board near ties, and
  445/445 reconstruction agreement (only 437/445 input recovery).
- [Linux gate](../2026-11-10-phase2-linux/README.md) was accepted first. The
  tokenizer assets/cases and artifact manifest are the same pinned ones; the
  [Phase 2 contract](../../../PHASE2.md) fixes the numerical budgets.
- The Python baseline is an independently qualified **adapted Python MLX port**,
  not unmodified upstream Apple execution. Its report is checked against the
  unchanged CUDA oracle before aggregation: 642 traces, 2,500 decisions and all
  declared budgets pass. The benchmark summary refuses a changed qualification,
  oracle policy, trace identity, source hash, artifact manifest, workload, output
  hash, trial matrix or tolerance check.
- [Summary](summary.json) and [raw reports](raw/) retain all 30 fresh-process
  reports (six codec plus 24 tokenizer), baseline report/qualification, commands,
  source/binary SHA-256s and CPU-activity observations. The compact raw JSON is
  about 3.2 MiB; it contains no weights or large trace arrays.
- Measurements ran on an M1 Ultra (64 GiB), macOS 26.7.3, Xcode 27.2, Rust 1.99.1,
  MLX 0.32.2 with the pinned project-local native MLX tuple. Raw environment data
  identifies exact package versions and binary hashes. The measurement worktree
  was based on `4668ce5` with the measured, uncommitted sources identified by
  SHA-256; the later publication commit records those sources without relabeling
  this historical run.
- Three fresh processes per runtime/workload were alternated Python/Rust/Python,
  then Rust/Python/Rust; each has five warmups and 30 timed trials. Tables pool
  90 trials/cell; p95 is nearest-rank. Medians, p95, extrema, standard deviation,
  IQR and per-process medians are in the summary. These are descriptive samples,
  not 90 independent-process confidence bounds.

The workstation was not exclusive. CPU activity was sampled every two seconds;
136 samples record Finder, indexing, desktop applications and other work. No
power/clock/tooling changes were made, and GPU temperature/utilization were not
available without privileged tooling. Warm filesystem cache is used; Python import
and MLX context startup are excluded. Small differences must not be overread.

## Matched codec execution

The fixed [workload](../../../tests/fixtures/codec-bench-v1.json) has 89-position
Sudoku boards, width 256 and twelve levels/322 codes at batches 1, 2 and 4.
Timing includes graph construction and synchronous MLX evaluation, starts from
host IDs and ends after host code/output-ID readback; validation/hashing and
object destruction are outside the timed region. Encode retains cumulative
latents; decode retains logits/IDs; reconstruct retains both. The resident
middle-block case has no host transfer. These positions are Sudoku tokens
including separators, not generated text tokens.

All runtime/process/scope input IDs, supplied codes and output IDs match. The
benchmark additionally compares 44,856 native logits against its qualified
same-device baseline using the unchanged `5e-4 + 2e-4 * abs(reference)` budget;
all pass. This supplements, rather than replaces, the full trace gate.

Wall median / p95 milliseconds (90 trials per cell):

| Batch | Scope | Rust MLX | Adapted Python MLX |
| ---: | --- | ---: | ---: |
| 1 | encode | 13.198 / 13.829 | 15.120 / 15.364 |
| 1 | decode | 11.888 / 12.066 | 13.693 / 14.510 |
| 1 | reconstruct | 24.791 / 25.041 | 28.417 / 29.105 |
| 1 | resident decoder middle block | 1.107 / 1.133 | 1.112 / 1.143 |
| 2 | encode | 13.959 / 14.132 | 15.935 / 16.160 |
| 2 | decode | 12.973 / 13.247 | 14.718 / 14.911 |
| 2 | reconstruct | 26.922 / 27.363 | 30.534 / 30.753 |
| 2 | resident decoder middle block | 1.184 / 1.213 | 1.179 / 1.229 |
| 4 | encode | 14.993 / 15.421 | 16.952 / 17.248 |
| 4 | decode | 14.151 / 14.324 | 15.879 / 16.022 |
| 4 | reconstruct | 29.040 / 29.943 | 32.652 / 32.986 |
| 4 | resident decoder middle block | 1.216 / 1.270 | 1.233 / 1.269 |

Reconstruction throughput Rust/adapted-Python is respectively **40.34/35.19**,
**74.29/65.50**, and **137.74/122.50** boards/s for B1/B2/B4 (3,590/3,132;
6,612/5,830; and 12,259/10,903 positions/s). This is an implementation comparison
on the same MLX backend, not a portable language or upstream speedup claim.
Separate synchronized stage barriers perturb execution, cannot be summed into the
headline, and include the final host-readback selection; MLX has no qualified
stream/kernel-only counterpart, so that field is explicitly null.

## Loading and memory

Both paths authenticate the same canonical artifact inventory and load 58 runtime
F32 tensors. The adapted Python path is a different source representation, so this
is not a parser-only comparison. Warm checked-load median is **943.60 ms Rust**
and **289.97 ms adapted Python**; first-load ranges are 943.61–945.38 and
280.42–294.33 ms. First B1 reconstruction ranges are 58.12–64.30 and
68.31–68.91 ms respectively.

The known native host-RSS weakness persists: final process HWM is 3.40–4.42 GiB
Rust versus 0.287–0.288 GiB Python. The separate host-only diagnostic is retained
at `reports/phase2-mlx-load-diagnostic`; it reproduces high native RSS. No loader
change was made merely to alter a measurement, and lower active MLX arrays do not
explain or excuse whole-process RSS. Native and Python MLX allocator observations
are similar in scope (B1 reconstruction peak roughly 105 MiB, active about 57 MiB,
cache about 49 MiB), but unified active/peak/cache is neither distinct VRAM nor
whole-process RSS. It is reported descriptively, not as an efficiency claim.

## Text tokenizers

The same 89 fixed Unicode/whitespace/special-token strings are encoded and their
fixed IDs decoded with and without special-token skipping. All first/warm outputs
and all cross-runtime/process framed hashes match. Native benchmarks run with a
cleared environment and no Python on `PATH`; the reference remains offline-only.
HF wrapper overhead/source representation means this is not a general language
benchmark.

Median milliseconds per complete 89-string pass (Rust/adapted Python):

| Profile | Encode | Decode | Skip-special decode |
| --- | ---: | ---: | ---: |
| BERT | 1.282 / 2.406 | 0.712 / 1.011 | 0.595 / 0.890 |
| GPT-2 | 1.351 / 2.474 | 0.175 / 0.504 | 0.184 / 0.514 |
| Qwen3 | 1.302 / 2.349 | 0.164 / 0.483 | 0.181 / 0.504 |
| TinyStories | 0.370 / 2.655 | 0.022 / 0.354 | 0.022 / 0.610 |

The summary separately labels strings/s, all-token positions/s and non-special
positions/s. Warm-load medians Rust/Python are BERT 11.81/17.50 ms, GPT-2
51.27/50.90 ms, Qwen3 183.64/144.38 ms and TinyStories 5.63/30.71 ms. Qwen3
native loading is slower; no universal load-speed claim is made.

## Validation and reproduction

The measured source exactly matched the separately checked Mac candidate files;
that MLX parity suite passed `./tools/check.sh --mlx`, including all existing
regressions and new codec tests. The final summary changes are covered by five
fail-closed report-tool tests; the MLX asset/reference tests pass with the pinned
MLX Python environment. No documentation-only publication edit is presented as a
fresh performance run.

From the repository root, with authenticated assets/native prefix and fresh output:

```sh
export PATH=/opt/homebrew/bin:$PATH
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
export RUSTDOCFLAGS="$RUSTFLAGS"
export CARGO_BUILD_JOBS=2
./tools/check.sh --mlx
cargo build --release -p ncm-backend-mlx --features mlx --example mlx_codec_bench
cargo build --release -p ncm-tokenizers --example asset_bench
python3 tools/bench/run_phase2_mlx.py "$PWD" reports/phase2-mlx-repeat
PYTHONPATH=tools/bench python3 tools/bench/phase2_mlx_summary.py \
  reports/phase2-mlx-repeat reports/phase2-mlx-repeat/summary.json
# Recheck this committed evidence without GPU/assets:
PYTHONPATH=tools/bench python3 tools/bench/phase2_mlx_summary.py \
  docs/results/2026-11-10-phase2-mlx/raw /tmp/phase2-mlx-summary.json
```

The final report keeps the negative RSS result, qualifying-baseline limitation,
workstation noise and non-comparable memory scopes visible. It makes no Phase 3,
generator, inference-release, losslessness, or full text-model claim.
