# Phase 2 MLX codec and tokenizer parity

**Correctness milestone, not the complete Apple runtime gate.** Linux acceptance
was completed/pushed in `fd1b482` before this implementation. SSH authentication
was restored by the user. The clean Mac checkout was fast-forwarded from Phase 1
to `4668ce5`; uncommitted candidates were tested in a separate disposable Git
worktree, never overlaid on the main checkout. Existing CPU work was left alone;
builds used two jobs. Acceptance performance/memory measurements are still pending.

## Implementation and checks

`crates/ncm-backend-mlx/src/codec.rs` implements the existing `WeightBackend` and
`CodecBackend` contracts using the same shared `ncm-model` codec. Opaque immutable
arrays prevent injected foreign-device graphs or storage mutation. Every operation
uses an explicit GPU stream, independent of the caller's scoped default; a real
GPU operation is evaluated during construction. No CPU fallback, Python production
path, unsafe Send/Sync implementation, dependency upgrade or model scheduling
copy was added. Existing MLX composed norms/resampling/attention are reused.

New tests cover dtype/shape rejection, first-index argmin/argmax including wide
rows/signed zeros, independently retained snapshots, batched transposed convolution,
full-span convolution/projection equivalence and restoration of a caller's CPU
stream. `evaluate` materializes supplied lazy roots synchronously. Uniform/typed
index uploads remain U32, with checked I64 conversion only in offline trace files.

Real M1 Ultra / MLX 0.32.2 results:

| Check | Result |
| --- | --- |
| Existing operator fixtures | All 18 unchanged cases pass |
| Components | 140 comparisons / 75,348 values; max abs `7.62939453125e-6` |
| Complete tiny codec | All 236 traces pass; snapshot/plain checks total 252 comparisons / 112,212 values; max abs `1.52587890625e-5` |
| Trained Sudoku | All 642 traces and 2,500 code/output decisions pass |
| Near ties | Same two mixed-board level 1/2 decisions, both IDs exact |
| Reconstruction vs Python | 445/445 positions, 5/5 sequences |
| Input recovery | 437/445 positions, including eight mixed-board losses |
| Tokenizers | 356 encode / 2,314 decode comparisons exact across four profiles; four asset mutations/profile rejected |

[Components](raw/components.txt), [trained capture metadata](raw/native.json),
[complete qualification](raw/qualification.json), [tokenizer results](raw/tokenizers.json).
The tokenizer probe also passes with a cleared environment and no Python on PATH.
Sudoku tokenizer unit regressions pass in the full local checks.

The trace executable now uses `tests/support/trained_codec.rs` for **both adapters**:
identical authenticated input/config/weight checks, independent retained observations,
fixed-code/free-running paths and uninstrumented equality checks. The CUDA harness
refactor was validated on Linux first: all 642 traces pass and the trace hash stays
exactly `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
No old CUDA report/source fingerprint was rewritten.

Two fresh MLX captures have identical metadata and trace SHA-256:
`f20ffe0f36a82c80a81e91a61b61950968821369b6b41d0f13cbf9e732875a0f`.
Raw arrays remain under `reports/phase2-codec/mlx-v2` and `mlx-v2-repeat` on the
Mac; the first capture is also copied locally. Durable large-trace publication
remains part of Phase 2 closure.

## Initial failure and native FP32 correction

The [initial capture](raw/initial-native.json) matched every selected ID but
[failed two floating values](raw/initial-qualification.json) in
`blank_batch1/encode.level_00.quant_input`. This was **not** accepted as parity:

| Channel | Pinned CUDA | Initial native convolution | Allowed absolute error |
| ---: | ---: | ---: | ---: |
| 113 | 4.466717243 | 4.468010902 | 0.0009933434 |
| 133 | -0.983386219 | -0.983037412 | 0.0002966772 |

The encoder outputs were exact and pre-quant outputs differed by at most
`4.76837158203125e-7`. This full-span convolution sums 22,784 products with
cancellation. Diagnostic host FP64 dot products from the respective already-FP32
pre-quant inputs were:

| Input source | Channel 113 | Channel 133 |
| --- | ---: | ---: |
| CUDA oracle | 4.467671124970701 | -0.983171347534181 |
| MLX capture | 4.467653622937439 | -0.983139918201196 |

These are diagnostics only, **not replacement goldens or production FP64**.
For zero-padding, kernel-length-equals-input-length convolutions, the adapter now
uses the mathematically equivalent native **FP32 matrix product**, flattening BCL
inputs against OIK weights. This is a generic shape case with exactly one output
position for any positive stride. Other convolutions stay native convolutions;
there is no custom kernel, checkpoint/channel special case or precision change.
An exact batched synthetic test covers strides 1/4/7; all existing tiny and trained
checks were repeated after the change. The original failure remains published.
[This reverse patch](raw/initial-convolution.patch) reconstructs the exact initial
adapter source (verified against its recorded SHA-256); apply only in a separate
checkout when reproducing the negative result.

The passing result is still close to the fixed budget: channel 113 is now
`4.467653751373291`, absolute error `0.0009365081787109375`, or **94.28%** of its
allowance. Maximum normalized errors for distance and accumulated categories are
4.91% and 12.73%. Full histograms, per-tensor errors and near ties are retained;
large distance magnitudes must not be judged by absolute error alone.

## Reproduction and remaining work

Use the unchanged [native MLX tuple](../../../agent-context-map.md#pins-machines-and-caches)
and the authenticated Sudoku bundle, trained CUDA oracle and canonical tokenizer
assets from Linux. From a checkout at this milestone:

```sh
export PATH=/opt/homebrew/bin:$PATH
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
export RUSTDOCFLAGS="$RUSTFLAGS"
export CARGO_BUILD_JOBS=2
./tools/check.sh --mlx
cargo test -p ncm-backend-mlx --features mlx --release --test codec_components -- --nocapture
cargo build -p ncm-backend-mlx --features mlx --release --example mlx_codec_trace
target/release/examples/mlx_codec_trace artifacts/canonical/sudoku-easy-qualified \
  reports/phase2-oracle/v1/sudoku-cuda reports/phase2-codec/mlx-new
cargo build -p ncm-tokenizers --release --example asset_probe
env -i PATH=/usr/bin:/bin target/release/examples/asset_probe \
  artifacts/canonical/tokenizers-qualified tests/fixtures/tokenizers-v1.json
# Offline qualification can run on Linux after copying the native capture:
.reference/venv/bin/python tools/reference/qualify_codec.py \
  reports/phase2-oracle/v1/sudoku-cuda reports/phase2-codec/mlx-new reports/phase2-codec/mlx-new-qualification.json
```

The actual scratch worktree was `.reference/phase2-mlx-candidate`, using the main
checkout's absolute native prefix and `CARGO_TARGET_DIR`. Its source hashes are
recorded in the capture. Checks passed through `tools/check.sh --cuda` on Linux
and `tools/check.sh --mlx` on the Mac, including Clippy, docs, all existing fixture
regressions and 11 report-tool tests. Native logs are local
`.reference/phase2-mlx-v2-check.log` and `.reference/phase2-mlx-v2-components.log`.

Next: independently qualify an adapted same-device Python baseline, implement
matched codec/tokenizer measurements, investigate the historical Apple load/RSS
weakness, and collect three fresh-process runs when competing work permits it.
These execution-affecting changes remain **performance-unqualified** until that
report exists. No Phase 3 work or overall Phase 2 completion is claimed.
