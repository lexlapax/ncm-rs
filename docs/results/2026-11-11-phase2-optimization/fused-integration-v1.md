# Approved strict-FP32 projection integration

The user approved the [v2 fused-projection patch](fused-projection-review-v1.md)
and requested continued alignment with the upstream Python algorithms. The
integrated backend still computes the same projection and bias, using strict
FP32 rather than cuBLASLt's TF32 default. Fusing the bias can change floating
point rounding, so numerical equivalence is determined by the existing pinned
trace and decision policy, not by a bitwise-equality claim. No upstream pin,
model or artifact schema, fixture, oracle or numerical tolerance changed.
The default backend trait implementation retains the original two operations;
CUDA uses the fused path only under its documented conditions. See the review
for its per-backend workspace, fallback, and remaining architecture limits.

The integrated source matches the approved v2 patch; the only addition is a
real-CUDA regression that enqueues an input write on a second stream before a
fused read. It passed eight rounds without an explicit foreign-stream wait;
this covers one dependency pattern, not all stream interleavings. The full
`umask 077; RUSTC_WRAPPER= CARGO_TARGET_DIR=... ./tools/check.sh --cuda` passed
before this test-only addition; the added test and CUDA Clippy passed afterward.
Fresh trained CUDA qualification passed **642 traces and 2,500 decisions**,
with zero ID disagreements and two near ties; all four real-tokenizer probes
passed. The integrated CUDA benchmark and trace binaries are byte-identical to
the independently built and qualified review binaries.

[Checks, qualification, tokenizer probes and binary hashes](raw/fused-integration-check-v1.tar.xz)
have SHA-256 `dd608c9dba6b941c4f3733bb1fadb49e508a4ee387849e113a2fce7c4b957271`.
The [separate cross-stream diagnostic](raw/fused-cross-stream-v2.tar.xz) retains
its source and output. The full trained capture stays local; its hash is in the
qualification report. These checks and the earlier matched diagnostic do not
constitute performance acceptance.

## Predeclared Linux gate

Measure the committed integrated source against accepted Rust `3307dff` and
the pinned Python reference: three fresh processes per runtime for all five
workloads, then an independent, unconditional nine-process-per-runtime control
for every workload and scope, followed by nine lifecycle processes. Use the
existing CPU 2 workload / CPU 0 observer placement and GPU/process telemetry;
keep all previous and new results, unchanged thresholds, and no selected reruns.
Audit the full control and source bindings. Apple work remains gated on a
passing Linux result.
