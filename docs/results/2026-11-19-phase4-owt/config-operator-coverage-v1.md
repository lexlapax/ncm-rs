# OWT config and synthetic operator coverage

Phase 4 requires synthetic, config and operator coverage for OpenWebText now. Checkpoint and
quality coverage wait until weights are released and validated. The pinned upstream repository has
`config/eval/owt.yaml`, but the pinned checkpoint revision still has only `owt/TODO.md`. OWT inference
remains unsupported and fails explicitly.

## Config

`ncm_core::OwtConfigOnly::from_json` validates the pinned configuration
([fixture](../../../tests/fixtures/owt-config-v1.json), converted from the upstream YAML with its hash
recorded). It checks:

- 32 levels on the squares ladder 1, 4, …, 1,024, with 8,192-entry codebooks (11,440 positions in total);
- a 16-layer generator with 16 heads and width 1,024 (heads of 64), context 1,025 and one prefix token;
- block-diagonal attention, RoPE with base 100,000, QK normalization, and level and position encodings;
- level zero from the checkpoint unigram (no prediction, no query, no context encoder);
- post-upsample convolutions with kernel 3, and codebook-tied heads with learned scale and bias;
- a GPT-2 tokenizer with `<|pad|>`;
- a codec with vocabulary 50,304, length 1,024, width 384, six heads, group norm 8 and a learned level-one cascade with base 4;
- temperature 1, top-k 0, top-p 1, CFG 0;
- `cut_at_eos: false`, the no-EOS-cut policy an implementation must preserve.

Changing any of these fields fails validation, and so does an empty document.
`require_released_checkpoint` always returns an explicit config-only error. `ncm-io` rejects OWT
artifact profiles in both full and inference-only loading.

## Synthetic operators

`crates/ncm-backend-cuda/tests/owt_shapes.rs` runs the shared backend operators on the CPU backend at
OWT's extreme shapes. It compares each with a naive host reference:

- area resampling of a 1,024-position latent to levels 1 and 961, and linear resampling back to 1,024 (max error 1e-6);
- split-half RoPE at 1,025 positions (prefix plus the 1,024-position top level), 16 heads of 64, base 100,000 (1e-6);
- masked attention at the same shape, with the prefix row unable to see targets; the first and last heads
  are checked against FP64 softmax attention on sampled rows (1e-5).

These tests use synthetic values, not OWT weights. They do not establish CUDA or Metal execution at these
lengths. The CUDA efficient attention kernel admits at most 129 positions and the fused rotary kernel at
most 560, so an OWT runtime would need longer-sequence kernels or the generic paths. Such a runtime needs
its own parity, quality and performance evidence once a checkpoint exists.
