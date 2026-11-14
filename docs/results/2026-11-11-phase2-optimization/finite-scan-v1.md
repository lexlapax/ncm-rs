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
