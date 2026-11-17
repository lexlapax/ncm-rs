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
