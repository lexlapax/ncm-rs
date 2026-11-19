# Phase 5 LM1B default rotary sidecar: CUDA screen

2026-11-19. The unified `ncm sample` command now defaults to the file `lm1b-reference-rope-v1.safetensors` **beside** the artifact directory; `--reference-rope FILE` overrides that location. The existing `sample-lm1b` alias still takes its original explicit rotary argument. This changes neither artifact contents nor manifest/schema: the sidecar is separately generated, and the existing validator still requires SHA-256 `9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`, exact length 32,920 and the pinned tensor shapes/types. Missing or incorrect files fail explicitly; there is no generated approximation or device fallback.

The table is derived from the pinned upstream rotary implementation, not model weights. On the Linux reference setup the existing offline export command writes the default location (which must not already exist):

```sh
setpriv --inh-caps=-all .reference/venv/bin/python tools/reference/lm1b_reference_rope_diagnostic.py \
  --upstream .reference/upstream-ncm \
  --output artifacts/canonical/lm1b-reference-rope-v1.safetensors
sha256sum artifacts/canonical/lm1b-reference-rope-v1.safetensors
```

For this screen the retained already-exported table was copied into that location; its hash matched. A numerical feasibility probe using Linux `libm` float32 `powf`, `cosf` and `sinf` in upstream operation order did **not** reproduce the pinned reference: 232 of 4,096 cosine values and 132 of 4,096 sine values differed, with maximum absolute error `5.960464477539063e-08`. No approximation was adopted. Exact-byte parity, not a tolerance change, is required.

On real CUDA, seed 42 and input `[[101]]`, all three sizes produced byte-identical stdout using the default sidecar, the explicit `--reference-rope` override, and the earlier retained baseline; this also held with `--decode`. Output hashes are in the [routing](cuda-routing-screen-v1.md) and [decode](cuda-decode-screen-v1.md) screens. The candidate release binary SHA-256 was `4b52d2fb565b7196b4027b70ad6389006a5fa571531c285c7225159bb43c98ae`; build/runtime environment and inputs match those screens. Negative controls with a missing default sidecar and a 32,920-byte all-zero sidecar returned errors identifying the missing path and wrong pinned hash, respectively, before artifact inference.

A later offline exporter safety change uses exclusive output creation, rejecting existing files and
both live and dangling output symlinks even if created after the initial check. Three offline tests
cover these cases. A new Linux reference export is byte-identical to the prior retained table and
matches the pinned hash; its ignored path is `reports/phase5-lm1b-rope-exclusive-v1.safetensors`.
This does not add GPU or quality evidence. Output parent directories must remain trusted and stable;
a write failure can leave a partial file for inspection.

CUDA CLI tests (11 passed, corpus test separately ignored), CUDA CLI all-target Clippy and `tools/check.sh --cuda` passed, including 18 real-device operator cases. The default sidecar convention, output equality, and setup remain unverified on Metal. Fresh-clone setup documentation, all remaining text-input decisions and the Phase 5 release gate remain pending. No quality or performance conclusion follows from this screen.
