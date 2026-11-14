# Grouped finite-value scan — provisional Linux candidate

Codec loading still validates the entire weight inventory, dtype, shapes and
finite FP32 values before the first backend allocation. The candidate tests the
IEEE-754 exponent of little-endian words in groups of 16 and checks the tail;
it accepts exactly finite bit patterns and needs no alignment or unsafe code.
Focused tests cover finite extremes, subnormals, signed zero, infinities and
NaN variants in every position around group boundaries and in an unaligned
buffer. Existing invalid-weight tests still require zero backend allocations.

An isolated CPU probe of the actual 59,586,048 tensor bytes measured 4.359 ms
for the previous scan and 2.995 ms for the grouped scan in one 30-trial run.
The [probe source](raw/finite-scan-probe-v1.rs) SHA-256 is
`7f1c2525e3e5a6bfb839f4e52f24cfcdba75f248f0d27f34d676dd2bbd229d70`.
This is a microbenchmark, not a load or acceptance result.

In a targeted CUDA diagnostic, three fresh processes per binary ran the existing
codec workload in rotating previous/candidate order on CPU 2, with five warmups
and thirty trials. The saved geometry candidate's checked-load process medians
were 192.200, 193.086 and 190.514 ms; the modified candidate's were 190.516,
189.226 and 189.188 ms. Medians of process medians were 192.200 and 189.226 ms
(ratio 0.9845). All twelve batch/scope output-ID hashes matched. Host load HWM
ranged 606–608 MB previous and about 604 MB candidate; three processes do not
establish a memory improvement. [Raw reports and binary hashes](raw/finite-scan-v1.tar.xz)
SHA-256: `c26300ecdc8ea8e0f05147ec0497094da5687b50db1a65e94ecf0d93d486bd52`.

These diagnostics are not a certified quiet-window run, a Python comparison,
or a 78-cell acceptance audit. The candidate is provisional: resource evidence
and the Linux acceptance gate remain necessary before Apple work or a keep claim.

`tools/check.sh --cuda` stopped before its CUDA section after one of 482 reporting
tests failed: `test_real_writers_and_owned_process_samples_compose` saw a
CPU-only child's `ObservationMismatch` on job 6; telemetry completed. Log
`/tmp/ncm-finite-scan-cuda-check-v1.log` SHA-256:
`3d6ec5a0a929cdddccdc1dcc6f74603611d3cc2f6123b2fe53ac2f0c0f2e25eb`.
The CUDA section then passed separately: CUDA Clippy, real-device backend tests
and all 18 operator cases. [Log](raw/finite-scan-cuda-direct-v1.log) SHA-256:
`49069b9ac66e4a139dc1267714e3f9c4961d86864a95457169fbcb304b2f2a74`;
[operator report](raw/finite-scan-cuda-operators-v1.json) SHA-256:
`c734b388dfd2527373f51f266cfcd2f054d4e23140578e655b68fd34c63857bc`.
The failed combined check remains failed. A fresh CUDA capture and the pinned
reference qualifier passed **642 traces and 2,500 decisions**, with zero ID
disagreements and two near ties. The native trace SHA-256 remains
`d180a52c011c8af15812d10941de8574b1671c2cd23de67e9abc828f7c5ecba4`.
[Qualification](raw/finite-scan-qualification-v1.json) SHA-256:
`a1e09d0e3dc1e1476e218c03b7de0abbe9fcb5254b0afe0245efd8a2a71436d6`;
[capture log](raw/finite-scan-trace-v1.log) SHA-256:
`b78d044938ba8fc439cfb756166cbb6df2183f25fc125d85ca4f763d13615300`.

The matched binaries and output paths are in the archive; each job used
`taskset -c 2 BINARY artifacts/canonical/sudoku-easy-qualified
tests/fixtures/codec-bench-v1.json NEW_REPORT`. Build used `RUSTC_WRAPPER=
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 cargo build
--release -p ncm-backend-cuda --features cuda --example codec_bench --locked`.
The trained capture used `taskset -c 2 target/release/examples/codec_trace
artifacts/canonical/sudoku-easy-qualified reports/phase2-oracle/v1/sudoku-cuda
/tmp/ncm-finite-scan-trace-v1 cuda`, followed by `.reference/venv/bin/python
tools/reference/qualify_codec.py reports/phase2-oracle/v1/sudoku-cuda
/tmp/ncm-finite-scan-trace-v1 /tmp/ncm-finite-scan-qualification-v1.json`.

## Fresh Linux performance gate

Candidate `a6290db` was measured against accepted Rust `3307dff` and pinned
Python using the existing three-process, five-warmup, thirty-trial matrix.
All **45 processes** completed. A separate lifecycle capture completed nine
fresh processes with matching outputs. The unchanged initial audit passed
**71/78** comparisons; seven comparisons against accepted Rust exceeded the
process envelope despite passing their point estimates. All Python comparisons
and all comparable memory cells passed.

One follow-up was declared before execution: nine fresh processes per runtime
for each of codec, BERT, Qwen and TinyStories, all scopes and checked loads,
with no retry or changed thresholds. The fixed plan SHA-256 is
`02b4f60a7b588a0a9d7b1ecbe504e3cb23bf6bcbbf65cfca55dbe0128b6479c4`.
All **108 processes** completed. The unchanged paired-process audit passes
**76/78**, with two comparisons still open:

| Comparison versus accepted Rust | Follow-up median ratio | p95 ratio | 95% upper bounds, median / p95 |
| --- | ---: | ---: | ---: |
| B1 resident decoder block | 1.01571 | **1.11340** | **1.11863 / 1.12152** |
| BERT checked load | 0.99984 | 1.00258 | 1.02067 / **1.34396** |

The limits remain 1.05 for median/memory and 1.10 for p95. Nine paired
processes leave broad intervals; that does not make either cell pass. The old
tokenizer controls cannot be bound by the existing audit because the runner's
hash changed after their collection, despite matching tokenizer binaries and
Python environment. They were not reused or relabeled. The initial and
controlled results, complete reports, telemetry, qualification, lifecycle,
plan, commands and logs are retained in the [archive](raw/finite-scan-linux-gate-v1.tar.xz),
SHA-256 `67d42d1ea4f28497e4771b3118de9f530af4ba212c7fc00ece69cd9653814907`.
Its controlled audit SHA-256 is
`668c4d3d656cb89c20c6286152fac751e7a54c06958e9bb37a454daa9ac98634`.

Before collection the GPU reported 0% utilization; a 15-second read-only
sample found CPU 2 median 9% and maximum 15% non-idle activity, and its SMT
sibling CPU 3 median 0% / maximum 1.01%. This is not a quiet-window certificate
or proof of conditions throughout the run. No system policy changed. To
reproduce the audit, extract the archive into an empty directory and run
`python3 /PATH/TO/REPO/tools/bench/audit_phase2_performance.py
reports/phase2-finite-scan-linux-gate-v1/controlled-config.json NEW_OUTPUT`;
it returns exit 1 for the two failures. This was reproduced byte-for-byte in
`/tmp/ncm-finite-gate-repro-v1`. The earlier 72/78 audit remains historical;
neither result is an Apple gate or Phase 2 completion.
