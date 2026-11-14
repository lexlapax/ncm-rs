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
qualification report. The independently measured v2 `codec_memory` executable
is also byte-identical to the integrated clean build (SHA-256
`67a41e25e1be9576f0b903f47c4d832af4af9ccf3bb0f685ee843106d0940ae7`),
binding its ten-cycle, 144-request workspace observations to this build. These
checks and the earlier matched diagnostic do not constitute performance acceptance.

## Predeclared Linux gate

Measure the committed integrated source against accepted Rust `3307dff` and
the pinned Python reference: three fresh processes per runtime for all five
workloads, then an independent, unconditional nine-process-per-runtime control
for every workload and scope, followed by nine lifecycle processes. Use the
existing CPU 2 workload / CPU 0 observer placement and GPU/process telemetry;
keep all previous and new results, unchanged thresholds, and no selected reruns.
Audit the full control and source bindings. Apple work remains gated on a
passing Linux result.

## Upstream alignment check

The local upstream checkout is clean at the declared `upstream.toml` commit
`c70274273f489abc73832b2b98584a8483b6b203`. Its `models/vae.py` decoder block
uses biased `nn.Linear` layers for `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`
and `mlp.c_proj`, with Q/K normalization, attention, GELU and residuals in the
same order represented by `crates/ncm-model/src/blocks.rs`. The validated
configuration restricts the initial codec to the supported architecture and
requires Q/K normalization; unsupported configurations are rejected. The new
hook changes how a projection plus bias is dispatched, not the model's layer
order or weight orientation. In particular, it does not substitute TF32.

The existing trace qualifier contains 24 outputs from those four decoder
projections across its trained cases; all pass their previously declared
numerical budgets. All 642 trace comparisons pass, as do all 2,500 recorded
decisions. The pinned reference capture uses inference mode, math SDPA and
TF32 disabled. These tests cover the pinned trained fixtures, not every possible
model or hardware shape, and fused bias need not be bitwise identical to
separate operations. The historical reference contract documents its known
CPU/CUDA difference separately; no oracle or tolerance was changed here.

For this inspection, `models/vae.py` SHA-256 is
`7d393b56813342278def3b053e83ec4ff4e56a492e4ddf2a2441e1c9f78d4317`;
`tools/reference/codec_trace.py` is
`f8703f7fed1327afc6dc5a085eff609dc11c6259d0ce14fceec6c9d68525f411`;
and the integrated CUDA module is
`0166e89caf3fa557cb8db44bd940314c0be9d0ddd2f5a3a9e1fcfcac45a17309`.
`git diff 3307dff 928894b -- upstream.toml tests/fixtures` is empty.
