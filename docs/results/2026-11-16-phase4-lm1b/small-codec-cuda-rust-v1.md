# Experimental LM1B-small CUDA codec

The opt-in `lm1b-experimental` codec uses the pinned checked artifact and the
shared codec implementation. The explicit profile covers an encoder without
positions, a four-block decoder with positional embeddings, and one learned
128-wide initial up/down convolution. Normal LM1B artifact construction still
rejects execution; this experimental loader is not model or release support.
The strict cuDNN convolution accepts the bounded 128-wide, stride-128,
zero-padding case required by this codec.

Two previously inspected development rows were compared against the pinned
upstream CUDA trace documented in [the reference record](small-codec-cuda-reference-v1.md).
Strict FP32 cuDNN passes all 362 tensor comparisons under the existing
`codec-fp32-v1` policy, including exact encoded codes and both decoded ID sets.
The candidate trace SHA-256 is
`26d5b1d0a3ced2b4e4a8f634f8ba54743cfda46407f36e5a499611fd4b13959d`;
the comparison report SHA-256 is
`9ff1fa60050a7d7a293f6589229a6746562907c8fa62893e16557ef6e8f0b277`.
An initial Candle CUDA capture had 120 policy failures, despite exact recorded
codes and IDs; it is retained. Enabling the cuDNN feature alone did not select
the strict backend and reproduced that same result. A first strict attempt
explicitly rejected the previously unsupported 128-wide convolution; the
bounded extension enabled the passing capture. No tolerance was changed.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_codec_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-codec-cuda-reference-v1 reports/phase4-lm1b-small-codec-rust-v5
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_lm1b_small_codec_trace.py reports/phase4-lm1b-small-codec-cuda-reference-v1/traces.safetensors reports/phase4-lm1b-small-codec-rust-v5/traces.safetensors reports/phase4-lm1b-small-codec-comparison-v2.json
RUSTC_WRAPPER= cargo test --locked -p ncm-core --test lm1b_small_config
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-core -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda
```

The comparison covers two inputs only, not generation, task quality,
performance, Metal, or the Phase 4 gate. All-feature model tests and all-target
lint for core, model and CUDA passed. The full CUDA check passed, including 18
device operator cases; the initial check stopped at formatting, corrected before
the passing run. Logs are in ignored `reports/`.

## Metal codec comparison

On the two pinned codec fixture rows, the Metal trace completes all 362
tensors. The unchanged comparison fails 47 checks. The level-zero decode
contribution exceeds the tolerance in two values, with maximum absolute
difference `0.00014495849609375`. The encoder level-zero quantizer input
also has two values outside tolerance, with maximum difference
`0.0067138671875`. These fixtures differ from the controlled stochastic
inputs, so this comparison does not by itself identify the cause of that
sample's first changed choice.

The Metal trace SHA-256 is
`08f3f4cb2751b900495130b85d3748f19ec620252ee26ba8bec05c7f7412c9a9`.
The failed comparison SHA-256 is
`5b0906269efbd893ae457354bf4246489dc0e775046ce8360c46ba50353fc045`.
The Mac trace and log are retained under
`reports/phase4-lm1b-small-codec-metal-v1/` and
`reports/phase4-lm1b-small-codec-metal-v1.log`.

The captured level-zero upsampled tensor is bit-identical between the
CUDA reference and Metal on these fixture rows. Their first recorded
level-zero decode contribution differences therefore arise in the
subsequent refinement operation. On that shared captured input, a
CPU FP32 convolution and residual calculation differs from both
captured outputs. Its maximum absolute differences are
`0.0001678466796875` from CUDA and `0.000091552734375` from Metal.
An FP64 CPU calculation also differs from both. These controls do not
establish a replacement implementation or identify every arithmetic
difference inside refinement.

The control checks the pinned artifact and both trace hashes. Its
source SHA-256 is
`d266baf8bed00074f1250ce0276f27680ac21d5c9fb6023eb810bf050f2c1ec2`
and its result SHA-256 is
`96954ed6accec215368943dbd21ec6eccfee0a89b69e84838ec0219704592983`.
Both are retained in `reports/phase4-lm1b-metal-refine-control-v1/`.

A strict PyTorch CUDA control applied the pinned level-zero weights,
bias and residual blend to the shared captured upsampled tensor. It
reproduced every CUDA reference contribution bit exactly. On the same
input and weights, the captured Metal contribution differs in 74,808
of 98,304 values, with maximum absolute difference
`0.00014495849609375` and RMS difference
`0.000004671209741979684`. This isolates the fixture's discrepancy
to the refinement computation on the two devices. It does not yet
supply a Metal correction.

The control source SHA-256 is
`00275d62bbc2786aed1ad7146592be9470b9e7ef28935c529bf73d1185d74939`
and its result SHA-256 is
`8fdb7209aa3c5b44c6ce18dc8361658b8adf1223ea6c13072e5758db2e749d28`.
They are retained under
`reports/phase4-lm1b-metal-refine-cuda-control-v1/`.

An isolated Metal experiment applied the existing XSum windowed
matrix-multiplication refinement to the LM1B length-128 shape. The
unchanged codec comparator reports 80 failing tensors, including 16
out-of-band level-zero decode-contribution values. The controlled
stochastic comparator reports 37 failing tensors, 483 differing codes
and 102 differing penalty-selected IDs. These counts describe separate
cascading continuations. The experiment does not resolve the existing
47 codec and 34 stochastic failures.

The reversible patch SHA-256 is
`783093c63e134d0efade2279d0a4ccd470b067d92513d5ab07e8426d1ca4d64f`.
The codec and stochastic comparison hashes are
`e1a15f08c04cfe71ab8f49f0430a531d0dbd6a93cb0397926151f0e7ac59cdba`
and `a36a357ab2ec5bd5b67821782aa7a26013f17dc84fa3cfe87a6b13841c1fa6fd`.
The patch and reports are retained under
`reports/phase4-lm1b-metal-refinement-experiment-v1/`. The Mac
checkout was verified clean after reversing the experiment.
