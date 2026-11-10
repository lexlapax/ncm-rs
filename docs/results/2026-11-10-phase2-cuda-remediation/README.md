# Phase 2 CUDA remediation: fewer launches and stable allocation

**Production CUDA improvements are qualified; the extended Linux gate is still
open.** All codec latency cells now meet the predeclared Python bands in the final
three-process comparison. Remaining measurement/uncertainty work is listed below;
Apple optimization and Phase 3 have not started. This is codec reconstruction,
not generation, official task quality, or a language-wide speed claim.

## Protocol and provenance

- RTX 3060 Ti, driver 615.78.08; same machine, pinned FP32 artifacts, oracle,
  inputs, output retention and eager Torch reference as the historical reports.
- Previous Rust: `0da7fbe`, saved release binaries, separate clean source worktree.
  Candidate: measured working-tree source hashes in each report (the pre-commit
  HEAD alone does **not** identify the candidate). `evidence.json` records final
  source/binary hashes and the published evidence inventory.
- [Acceptance manifest](../2026-11-10-phase2-performance-research/acceptance-v1.json)
  was written before production changes: maximum median/p95 ratios 1.05/1.10,
  comparable host peak/live memory 1.05; compare against previous Rust and Python.
  Uncertainty cannot be resolved by widening a band or hiding a failed cell.
- Three fresh processes per runtime; five warmups and thirty measured trials per
  cell. The new runner rotates previous/Python/candidate order between triplets.
  A Python report is reused by both pairwise summaries, not independent evidence.
- User authorized stopping `llama-server.service` (stopped, not disabled) and
  closed Chrome. Hyprland remained running; no system clocks, power policy, global
  tools, or unrelated applications were changed. The final capture contains 209
  one-second GPU telemetry samples and process-utilization samples. Observed SM
  activity was attributed to the benchmark processes, but sampling is not proof
  that the desktop was inactive. SM clocks ranged 210–1965 MHz; no fixed-clock claim.
- Warm filesystem cache, not disk-cold. Context/import startup and output
  destruction remain outside the headline scopes. First loads and first execution
  are separate. CUDA stream spans include submission gaps, not just kernel time.

### Scope audit

Both codec paths authenticate all declared payloads, retain the required independent
latent snapshots, synchronize outputs, and transfer public IDs. The resident block
retains its device output but excludes host copies. Validation/hashing and destruction
are outside execution timing. Python encode's unused upstream `z_levels` work is
**not** silently deleted. Python decode retains the already-documented independent-
snapshot reference fix. Native upload contains 58 inference tensors; the original
Python module also keeps training buffers. These are implementation/API comparisons,
not identical instruction streams. Tokenizer calls retain all 89 scalar outputs,
check both skip modes, and keep validation/destruction outside timing. Python source
assets and Rust canonical assets differ in format; full checked loads are not a
format-controlled parser microbenchmark. No checksum, preflight, path check, schema,
precision, golden or tolerance was weakened.

## Research and mechanisms

The [initial research](../2026-11-10-phase2-performance-research/README.md) documents
pinned NCM/Torch interpolation and rejects Candle's FP64-sample bilinear shortcut.
Additional sources inspected on 2026-11-10:

- Candle `candle-core`, `candle-kernels`, and `candle-nn` **0.11.1**, registry source
  corresponding to commit `59f8e903ca80a32ea7f8514b39748fc979e69e82`; especially
  `candle-nn/src/ops.rs`, CUDA normalization/softmax kernels and tensor layout paths.
  [Source](https://github.com/huggingface/candle/tree/59f8e903ca80a32ea7f8514b39748fc979e69e82).
  MIT/Apache-2.0 dependency, not copied custom kernel code.
- cudarc **0.19.10** context/device ownership and stream-ordered allocation source.
  The current process's default CUDA pool is shared by callers using that pool;
  it is not a private allocator owned by each `CandleBackend` instance.
- NVIDIA [stream-ordered memory allocator](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)
  documentation (download identifies Driver API 13.5), particularly the release
  threshold and synchronization behavior. This does not change the installed
  CUDA 13.4 build toolchain. Zero is the eager-release default; retention is a
  threshold, not preallocation or a hard capacity/retained-cache guarantee.
- Installed NVIDIA CUPTI `cupti_trace_injection` sample and activity API docs:
  [CUPTI](https://docs.nvidia.com/cupti/main/main.html). Built locally under
  `.reference/phase2-cupti`, without installing Nsight or changing global tooling.
  NVIDIA sample source/license remains in the installed toolkit, not vendored.

### Changes, in order

1. **Vectorized FP32 resampling:** shape-derived gathers/weights for half-pixel
   linear interpolation; padded bin gathers/masks/reduction for adaptive area.
   Tensor values stay on device. Input is explicitly made contiguous before
   `index_select`; an initial strided-input failure is retained in the research
   history. Expanded gather storage and metadata uploads are included in measured
   costs. Existing composed operators remain the diagnostic/reference path.
2. **Existing fused CUDA primitives:** add optional, exactly pinned
   `candle-nn = 0.11.1` for its FP32 layer norm, RMS norm and softmax. This is a
   deliberate same-runtime dependency addition, not a backend/version upgrade.
   CPU diagnostic compositions remain; no custom kernel or precision substitution.
3. **Preserve immutable views:** `permute`/`narrow` no longer force eager copies.
   Dense-only consumers materialize explicitly. Snapshot independence, shape checks,
   stable first-index/signed-zero selection and readback contracts remain intact.
4. **Pool policy:** `CandleBackend::cuda` sets a 128 MiB release threshold. The
   explicit `cuda_with_pool_retention(ordinal, bytes)` API allows zero/eager release
   or caller-selected retention. This changes the per-device default pool **in the
   current process**, including other libraries sharing it; callers must coordinate
   policy. No system-wide setting changes. The small unsafe shim binds the owned
   context, uses its actual `CUdevice`, passes a live `u64`, checks both driver
   results, and never destroys CUDA-owned handles. A real-CUDA regression test
   covers default, clone, opt-out and custom settings.

Selection remains the checked value-reduction/masked-index-reduction implementation.
No evidence justifies replacing it with a custom kernel now that full codec cells
meet the Python latency bands. Loading/tokenizer production code is unchanged.

### Causal profiles, including failures

`codec_profile` performs checked load plus exactly two B1 reconstructions. CUPTI
whole-workload kernel counts fell **16,756 → 1,404 → 1,312** for previous,
vectorized-only, and final code. Activity counts include load/first execution;
summed durations overlap and are **not** acceptance wall times. The parser retains
kernel/API names, copies, allocation events and input digests, and rejects captures
reporting dropped records. Raw captures are archived; no dropped-record messages
were observed.

Vectorization/fusion alone did not fix B4 resident-block bimodality. CPU affinity
alone also failed. Two matched `--profile-block4` diagnostics had nearly identical
972 kernels and ~10 ms summed kernel duration, but `cuMemAllocAsync` totals were
**0.636 versus 20.497 ms**. The slow capture contained 35 allocations over 100 µs.
Its pool shrank to 64 MiB from a 96 MiB peak; the fast capture retained 96 MiB.
With retention, the matched diagnostic used **0.642 ms**, with no allocation over
100 µs. Instrumented diagnostics are marked and explicitly rejected by the timing
summary. They demonstrate the mechanism, not speedup measurements.

The earlier `vectorized-v1` matrix remains published: B4 resident block was
0.814 ms versus Python 0.317 ms, so that candidate did **not** meet acceptance.
`threeway-v1` is also retained; `threeway-v2` follows the final device-handle safety
cleanup. No slower run was deleted. Intermediate candidates were working-tree
experiments, not separately released revisions; their reports retain source hashes.

## Final three-way latency

Milliseconds, pooled medians; each distribution has three independent process
medians in the linked summaries. These are descriptive, not a statistical speedup
guarantee. All twelve codec medians and p95s meet the Python limits; the largest
median ratio is B4 block, approximately 1.006. Both full comparison rounds remain
visible: B1 reconstruction varied from 2.970 ms in v1 to 3.285 ms in v2.

| Batch | Scope | Previous Rust | Candidate Rust | Python |
| --- | --- | ---: | ---: | ---: |
| 1 | encode | 10.257 | 2.063 | 3.563 |
| 1 | decode | 8.011 | 1.244 | 1.521 |
| 1 | reconstruct | 18.165 | 3.285 | 4.659 |
| 1 | resident block | 0.308 | 0.178 | 0.253 |
| 2 | encode | 14.522 | 2.402 | 3.471 |
| 2 | decode | 12.117 | 1.619 | 1.739 |
| 2 | reconstruct | 26.681 | 4.003 | 5.081 |
| 2 | resident block | 0.402 | 0.233 | 0.281 |
| 4 | encode | 15.735 | 3.020 | 3.883 |
| 4 | decode | 13.329 | 2.131 | 2.128 |
| 4 | reconstruct | 28.949 | 5.134 | 5.859 |
| 4 | resident block | 0.841 | 0.315 | 0.313 |

Reconstruction is about **5.5–6.7× faster than previous Rust** in the final round,
without changing the shared model or precision. Warm codec load is
195.88/194.45/286.25 ms (previous/candidate/Python). Candidate process-first checked
load is 202.90–204.06 ms; first B1 execution 55.31–56.07 ms. These exclude process/
import/context startup and must not be called full cold-start latency.

### Memory and tokenizers

Final candidate host HWM is **577.4–580.7 MiB**, versus previous Rust
577.9–580.0 MiB and Python 1544.1–1565.5 MiB. Candidate final RSS is
558.2–561.3 MiB. Loading and host-memory advantages are preserved; no loader
optimization is claimed.

At B4, native live default-pool usage after output drop is 57.17 MiB. Reconstruction
pool peak is 72.82 MiB; current/peak reservation is 96/96 MiB. Before retention,
reservation could fall to 64 MiB between calls: **up to 32 MiB more reusable device
reservation is intentionally kept for this workload**, without that being live
model data. After load/drop loops, live pool usage returns to zero, while 64 MiB
reservation remains. Torch B4 reconstruction reports 107.11 MiB allocated peak
and 144 MiB reserved; these counters have different coverage and are **not**
comparable process/device peaks. Device-global observations include the desktop.
The 128 MiB policy is not a promise about larger, unmeasured workloads.

All tokenizer cells meet the Python latency bands. Qwen warm load is
164.98/164.26 ms, a 0.44% native excess; BERT/GPT-2/TinyStories are faster. Native
old/candidate `asset_bench` binaries are **byte-identical**. Nevertheless, original
uncontrolled comparisons contain failing old-Rust TinyStories cells: v1 decode
p95 ratio 1.419 and skip-special median 1.067; v2 skip-special median 1.059.
They are retained, not waived because Python is much slower.

A fixed follow-up collected **nine additional process triplets**, same unchanged
work and CPU-2 affinity for all three runtimes. Candidate/previous median ratios
for encode/decode/skip-special are 1.0019/0.9886/1.0183; p95 ratios
1.0021/1.0199/1.0181. However, the descriptive paired whole-process bootstrap's
skip-special median interval is **[0.9772, 1.0551]**, crossing the 1.05 ceiling.
This is unresolved measurement uncertainty, not evidence of a tokenizer code
regression or a passing extended gate. See `tiny-control-summary.json`; no
individual trials are treated as independent processes.

## Correctness and checks

- `./tools/check.sh --cuda` passes real-CUDA components, complete tiny codec,
  stable ties/snapshots, the new resampling/pool tests, all 18 fixed operators,
  CPU/default checks, formatting, Clippy and docs. Log: `raw/cuda-check.log`.
- All **642 trained traces** and **2,500 decisions** pass the unchanged budgets;
  zero ID disagreement, same two near ties. Reconstruction agreement is 445/445,
  input recovery 437/445, not losslessness or Sudoku-solving quality.
- Broader synthetic CUDA resampling passes **360 rows / 753,984 values** against
  pinned **Torch CUDA**, including strided inputs and both directions. Maximum
  absolute error is 5.2453e-5 and maximum budget fraction 0.628. This is not the
  much smaller Torch-CPU comparison error. The FP64-coordinate alternative still
  fails: 70 rows / 228 values, maximum budget fraction 7.107; retained separately.
- Benchmark summary checks pass exact output hashes/counts and all retained logits.
  The full trained gate is independent and is not replaced by benchmark outputs.
- Final `./tools/check.sh` passes, including **20 report tests**; **21 offline
  reference tests** pass. A clean staged checkout without ignored files reproduces
  all six matrix summaries (only historical summarizer provenance differs), six
  CUPTI summaries and the control summary; all evidence/source fingerprints match.
  Profile/control parsers have mutation/regression tests. No Python is introduced
  into production execution.

## Reproduce from committed evidence

Archives contain only benchmark JSON/telemetry, diagnostic profiles and logs—not
weights, training data, executables, or the large model-trace archive. Extract to
new directories. All prior raw reports and both final rounds remain immutable.

```sh
mkdir -p reports/remediation-recheck reports/remediation-tiny-recheck
r=docs/results/2026-11-10-phase2-cuda-remediation
tar -xzf "$r/raw/threeway-v2.tar.gz" -C reports/remediation-recheck
python3 tools/bench/phase2_summary.py reports/remediation-recheck/candidate \
  reports/remediation-recheck/candidate-rechecked.json
python3 tools/bench/phase2_summary.py reports/remediation-recheck/previous \
  reports/remediation-recheck/previous-rechecked.json
tar -xzf "$r/raw/tiny-control-v1.tar.gz" -C reports/remediation-tiny-recheck
python3 tools/bench/summarize_tokenizer_control.py reports/remediation-tiny-recheck \
  reports/remediation-tiny-recheck/rechecked.json
# Compare against the committed summaries. No GPU/weights/ignored baseline needed.
```

Live commands (fresh output paths; build binaries first):

```sh
export NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4
./tools/check.sh --cuda
cargo build --release --locked -p ncm-backend-cuda --features cuda \
  --example codec_bench --example codec_trace --example codec_profile
cargo build --release --locked -p ncm-tokenizers --example asset_bench
# OLD_CHECKOUT must be clean 0da7fbe; OLD_BIN holds its separately built binaries.
python3 tools/bench/run_phase2_linux_comparison.py "$OLD_CHECKOUT" "$OLD_BIN" reports/new-comparison
python3 tools/bench/run_phase2_linux_comparison.py "$OLD_CHECKOUT" "$OLD_BIN" \
  reports/new-tiny-control --repeats 9 --workloads tinystories --cpu 2
CUDA_INJECTION64_PATH="$PWD/.reference/phase2-cupti/libcupti_trace_injection.so" \
  target/release/examples/codec_profile artifacts/canonical/sudoku-easy-qualified \
  > reports/new-cupti.txt 2>&1
python3 tools/bench/summarize_cupti_trace.py reports/new-cupti.txt reports/new-cupti-summary.json
```

A checked local rebuild recipe for the installed sample (output stays local):

```sh
NVCC_CCBIN=/usr/bin/g++-15 make -C /opt/cuda/extras/CUPTI/samples/cupti_trace_injection \
  CUDA_INSTALL_PATH=/opt/cuda LIBNAME="$PWD/.reference/phase2-cupti/rebuilt.so"
```

The historical injection-library digest and installed sample source hashes are in
`evidence.json`; the original compiler invocation was not retained, and the checked
rebuild has a different binary digest. Do not infer instruction-identical profiler
overhead or use profiled runs as acceptance measurements. Benchmark argv, working
directories, package versions, source/binary hashes, timestamps, telemetry and GPU
activity are in the archives.

## Still pending; no phase advancement

1. Resolve TinyStories process-level uncertainty without relaxing thresholds or
   deleting negative cells; finish a fail-closed, whole-matrix acceptance audit.
2. Add the separately labeled process/import/context-startup and teardown evidence
   required by the extension; current first-call reports are not a replacement.
3. Complete formal extended Linux acceptance; committed-only report regeneration
   has been checked, but it does not settle the remaining hardware/scope questions.
4. Only then recheck Mac availability, safely pull reviewed pushed commits and
   diagnose/fix Apple loading, host allocation amplification and tokenizer loading.
   Any later shared change returns through Linux first. Phase 3 remains blocked.
