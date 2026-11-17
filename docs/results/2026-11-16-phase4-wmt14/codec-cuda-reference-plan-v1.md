# WMT14 CUDA codec reference plan

The WMT14 artifact and host-only codec/generator preflights are documented in the
[artifact record](artifact-plan-v1.md). Before enabling codec execution, capture
an upstream Python CUDA reference using the pinned checkpoint revision and clean
pinned source. Use the first two independently verified WMT14 source-tokenizer
ID rows, truncate/pad each to the codec's 64-token context with PAD 151643, and
run the upstream VQVAE in evaluation mode. Preserve the existing codec trace
instrumentation and numerical policies; do not modify upstream math or change a
tolerance. Disable TF32 and use the math SDPA backend, and record device,
driver, source, checkpoint, configuration, input and trace identities.

The two-row trace is a development fixture for encoder, quantizer and decoder
comparison, including forced-code decoding. It is not full WMT14 quality,
generator, performance or Metal evidence. Any unsupported operation or numeric
failure remains explicit; no production execution will be accepted on the basis
of artifact loading alone.

## Captured reference

Two independent CUDA runs completed using the following command, changing the
output suffix from `v1` to `v2` for the repeat:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_codec_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --reference reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-codec-cuda-reference-v1
```

Both reports have SHA-256 `164dc6b368dd71f6b6541a99aa50c2c514b3349a6fb6b2e787b3d5a5fbeef586`;
both 202-tensor payloads have SHA-256
`b4b66ab0b58910e501aed704d50df9493bad65d691e78738f8689d14649662fd`.
The capture script has SHA-256
`6cbc47858a42d4c68a715d141f0703cfe5d2dedde850d21fd3a55879ea141eb8`.
Logs and payloads remain under ignored `reports/`. The focused input validation
test passed with `python -m unittest discover -s tools/reference -p
'test_wmt14_codec_trace.py'` in the pinned reference environment. These
repeatable upstream outputs are a fixture, not a Rust comparison or a GPU gate.

## Experimental Rust CUDA comparison

The non-default `wmt14-experimental` feature exposes a development-only codec
loader. It validates the WMT14 artifact profile, task, model, tokenizer and
codec configuration, then preflights all 63 codec tensors before device upload.
The normal artifact loader still rejects WMT14. A two-row real-CUDA capture
used the exact manifest and reference hashes checked by the example:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= CUDA_COMPUTE_CAP=86 cargo build --release -p ncm-backend-cuda --features wmt14-experimental --example wmt14_codec_trace
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_codec_trace artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-codec-cuda-reference-v1 reports/phase4-wmt14-codec-rust-cuda-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_codec_trace.py reports/phase4-wmt14-codec-cuda-reference-v1/traces.safetensors reports/phase4-wmt14-codec-rust-cuda-v1/traces.safetensors reports/phase4-wmt14-codec-cuda-comparison-v1.json
```

All 202 tensors pass the unchanged policy, including exact recorded code and
decoded IDs. The largest absolute difference is 0.001953125 in
`encode.level_10.distances`, within its existing distance band. The native
capture SHA-256 is `3dca1fda4870448aaf832aa7487ecbfb3e101e5693f10afc994633483e48c009`;
the comparison SHA-256 is `50850853e742b501de21ba8b4337704ff81e2061e3e8516b54c9b182f48635bd`;
and the executable SHA-256 is `378e16203d7cb0a82dfe5c85ce196f9914f8bf38566edbf177648ee39f9975e4`.
The native capture checks fixed input and artifact hashes but did not perform a
source postflight. Desktop GPU processes were present. The first sandboxed
build failed GPU capability detection; the explicitly targeted rebuild passed.
Comparator band/nonfinite/ID regression tests and feature-specific Clippy pass.
Feature-enabled model tests and the full `tools/check.sh --cuda` regression pass;
the latter exercised 18 real CUDA operator cases (log SHA-256
`fbcb58a2d49802b6afcab5bc3e7d8745d0b8673069fab744ed6ba5c79b09e245`).
These two inputs do not qualify the generator, task quality, performance or
Metal, and do not resolve the separate XSum Metal raw-ID failure.
