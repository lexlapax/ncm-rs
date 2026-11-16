# WMT14 checkpoint inventory preparation

On 2026-11-16, Linux CPU inspection of the existing WMT14 checkpoints used
`.reference/venv/bin/python tools/reference/wmt14_inventory.py artifacts/upstream/wmt14-phase4 > reports/phase4-wmt14-inventory-prep-v2.json`
The script checks the previously documented checkpoint
sizes and SHA-256 values before `torch.load(weights_only=True, mmap=True)`; it also
binds the locally available config (2,177 bytes, SHA-256
`6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72`).
The config hash is an observed local identity, not a separately verified upstream pin.

The generator has 186 tensors, 480,523,968 tensor bytes: 185 FP32 and one
I64 (`lvl_1L`, shape `[1, 312]`). The tokenizer checkpoint has 96 tensors,
645,757,952 tensor bytes: 85 FP32 and 11 I64. The existing TinyStories training
name pattern matches 33 tokenizer entries; classification for WMT14 still needs
verification against upstream use. The config specifies 96 prefix tokens, 64
output positions, a 151,680-token codec vocabulary and eleven codec levels.
The JSON SHA-256 is `78564ad120432623d872e29e7eafa181a4e15c95201edf73ac390a0d6abb088b`; it lists all names, shapes and dtypes; it is ignored local evidence, not
an export or an inference validation. Input verification precedes inspection;
there is no device upload. WMT14 implementation, profile bounds, tensor
classification, tokenizer parity, quality and CUDA/Metal gates remain open.

The checked-out upstream source is at the existing pinned commit
`c70274273f489abc73832b2b98584a8483b6b203`. Its `scripts/eval_shard.py`
encodes source strings with the configured Hugging Face tokenizer, truncates to
96 IDs and pads to 96; the configured default has no frequency or adjacency
penalties, so final tokens use argmax. It cuts at EOS and filters EOS/PAD before
special-token-skipping decode. `scripts/eval_merge.py` reports cased and lowercase
corpus sacreBLEU, chrF and whitespace-token length ratio. These source observations
are preparation for an evaluator, not a qualified metric implementation or a claim
that other generation settings are supported.

For further classification, upstream `models/quant.py` initializes EMA and hit
buffers for every level; EMA updates are guarded by training mode, while hit
buffers can be updated in the quantizer's general forward path. The generated-code
`idx_to_zhats` path does not directly access these buffers. A complete call-graph
and artifact-export review is still needed before classifying all 33 entries as
training-only for every supported operation.

Comparing inventory names with the existing XSum v2 manifest found WMT14's
`lm_head_l0.weight` and ten `lm_heads.*.weight` entries, while XSum instead has
`l0_queries` and nine projection/temperature pairs. The names alone do not prove
semantics; they establish that the XSum head layout cannot simply be assumed
for WMT14. The full WMT14 tokenizer checkpoint includes all 53 names in XSum's
runtime codec manifest plus 43 additional names, including the 33 patterned
EMA/hit buffers and extra level-specific tensors; shapes also require independent
validation.

## Full source-tokenizer corpus screen

The pinned local Transformers tokenizer capture was generated with
`.reference/tokenizers-venv/bin/python tools/reference/wmt14_tokenizer_inputs.py --output reports/phase4-wmt14-tokenizer-reference-v1.json`.
The capture verifies the dataset hash, pinned tokenizer manifest, source-file hashes
and tokenizer library versions; it uses local files only. Its SHA-256 is
`22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f`.
The first attempt failed before capture because an unrelated helper imported Torch
in the tokenizer-only environment; the helper was replaced by a local hash check.

`RUSTC_WRAPPER= cargo run --release --locked -p ncm-tokenizers --example wmt14_corpus_probe -- artifacts/canonical/tokenizers-qualified/qwen3 artifacts/canonical/text-phase4-v1/wmt14.json reports/phase4-wmt14-tokenizer-reference-v1.json`
matched the complete encoded ID sequences for all 3,003 source rows (108,807 IDs)
and every 96-token truncated/padded prefix; 24 rows were truncated. The focused
strict Clippy check passed. Log: `reports/phase4-wmt14-tokenizer-corpus-v1.log`.
This establishes only source-side tokenization for these inputs; decoder parity,
model execution and BLEU remain unqualified.

In the pinned generator, omitted `query_level0` defaults to false. With WMT14's
`predict_level0: true`, generation first runs the prefix through the transformer,
selects the position indexed by the non-padding token count minus one (clamped
to the valid prefix range) for this one-token first level, and uses `lm_head_l0`; subsequent levels use the other heads. This follows upstream
`generate` and `_level0_slice` source inspection and still requires trace validation.

`tools/reference/wmt14_artifact.py` now performs a preparatory structural
preflight: it preserves all 186 generator tensors, separates the exact 33
EMA/hit-named tokenizer tensors from the other 63, and rejects malformed
metadata, renamed buffers and shared storage. The helper does not export an
artifact or settle inference classification. The five focused inventory and
preflight tests passed under `.reference/venv/bin/python`; a real CPU-only,
`weights_only=True, mmap=True` load produced the expected 186/63/33 counts.
An attempted inventory test with system Python failed because it lacked Torch;
the existing reference virtual environment passed.
