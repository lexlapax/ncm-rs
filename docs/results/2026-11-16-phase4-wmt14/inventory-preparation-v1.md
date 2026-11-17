# WMT14 checkpoint inventory preparation

On 2026-11-16, Linux CPU inspection of the existing WMT14 checkpoints used
`.reference/venv/bin/python tools/reference/wmt14_inventory.py artifacts/upstream/wmt14-phase4 > reports/phase4-wmt14-inventory-prep-v2.json`
The script checks the previously documented checkpoint
sizes and SHA-256 values before `torch.load(weights_only=True, mmap=True)`; it also
binds the locally available config (2,177 bytes, SHA-256
`6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72`).
At the initial inventory, the config hash was a local observation; subsequent
verification against the pinned checkpoint revision is recorded below.

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

For the pinned `eval_shard.py` generation path specifically, `run` calls
`net.generate`, reconstructs the generated indices with `idx_to_zhats`, and
passes the resulting latent to the decoder. The generator uses the codebook
embeddings and `get_next_autoregressive_input`; those helpers and
`idx_to_zhats` do not access the EMA or hit buffers. The alternate VQ-VAE
`forward`/`reconstruct` path calls the quantizer's general `forward`, which
does update hit buffers. This narrows the classification for the evaluated
generation path only; it does not authorize dropping tensors for other APIs or
replace artifact and trace validation.

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

The later decoding screen constructed two synthetic sequences from each of the
3,003 verified source-ID rows: a 64-token prefix and a variant with PAD, a
special token, EOS and trailing data. The pinned local Transformers tokenizer
captured the evaluator's EOS truncation, PAD filtering and special-token-skipping
decode via `setpriv --inh-caps=-all env TOKENIZERS_PARALLELISM=false
.reference/tokenizers-venv/bin/python tools/reference/wmt14_decode_reference.py
--output reports/phase4-wmt14-decode-reference-v1.json`; that reference hashes
to `d38eaa32c1d92e55d421f524bae456765bb64e9a06f2a75fa2ba5f0f73081c98`.
`setpriv --inh-caps=-all env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 cargo run
--release --locked -p ncm-tokenizers --example wmt14_decode_probe --
artifacts/canonical/tokenizers-qualified/qwen3
reports/phase4-wmt14-decode-reference-v1.json` matched all 6,006 decoded texts;
focused strict Clippy passed. Reference and probe source SHA-256 values are
`ceb3394e7b5ad2fe33f079646ea16db22b0f1dbab296909da43c3d41516d1f71` and
`20b7bdecbfae8d76fae88c8915d63b01350aa38da7afc6fb59f6b1765f3dc933`.
These synthetic source-derived sequences do not qualify generated translations,
model execution, BLEU, or all possible token sequences.

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

## Experimental lossless export

The separate `wmt14-preparation-v1` inventory preserves all 186 generator and
96 tokenizer checkpoint tensors, including every EMA/hit buffer. It is not a
runtime artifact or an inference qualification. The CPU-only command was:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_export_preparation.py --source artifacts/upstream/wmt14-phase4 --tokenizer-asset artifacts/canonical/tokenizers-qualified/qwen3 --output artifacts/canonical/wmt14-preparation-v1
```

Each serialized tensor was reloaded and compared for name, dtype, shape and
exact bytes. The exporter checks the checkpoint and locally observed config
hashes and the pinned tokenizer inputs before and after export; it does not
itself independently verify the local config against upstream. Its
manifest SHA-256 is `3653e4c5bede7b0d8b89148ecccf580ac5cc87617697b737582e8173586334de`;
the exporter SHA-256 is `10f6eacdb97eb7f1a3e282f3dab06728a72684d148c24382291956a0f5480c28`.
Seven focused WMT14 tests passed, including bit-preserving roundtrip and
existing-output rejection (`reports/phase4-wmt14-export-preparation-tests-v1.log`).
The full runtime schema, implementation and GPU comparisons remain pending.

The config was subsequently fetched over HTTPS from the existing checkpoint
revision at [wmt14/config.yaml](https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692/wmt14/config.yaml)
using `curl -fsSL --max-time 20 https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692/wmt14/config.yaml -o
reports/phase4-wmt14-upstream-config-v1.yaml`. `cmp` and SHA-256 confirmed the
download and local config are byte-identical at the hash above. The first
sandboxed fetch failed DNS resolution; the authorized network-enabled read
succeeded. This verifies the config at the existing revision without changing
the upstream revision or an accepted runtime artifact.

Comparing this preparation manifest with the pinned XSum v2 manifest gives
175 common generator names (three shape differences), 11 WMT14-only and 19
XSum-only names. The shared level and mask tensors reflect WMT14's 312 positions
after level zero; its ten level embeddings differ from XSum's nine. The 53 shared codec
names include five shape differences, including the larger token embedding/output
vocabulary and two learned-resampling kernels. This inventory comparison narrows
the implementation work but does not establish equivalent layer semantics.

## Single-row upstream CPU probe

Two fresh CPU runs loaded the pinned upstream model and both checkpoint files
strictly, used the first previously verified 96-token prefix, and captured 313
generated codes and 64 argmax decoder IDs. The command was
`setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/reference/wmt14_cpu_probe.py --upstream .reference/upstream-ncm --source
artifacts/upstream/wmt14-phase4 --prefixes
reports/phase4-wmt14-tokenizer-reference-v1.json --output NEW_REPORT`.
Both output files are byte-identical, SHA-256
`3bbf94a1509151fd4351309348cab4bdbd2c748f728e162b7b3c708f520c5c5b`;
the probe source SHA-256 is
`a97b6041aa0d4b9dca97b50167b7ac8cdcc377017b578e121c3dab7d9b09125b`.
The probe checks the observed generation configuration, source and checkpoint
identities and output dimensions. It is a narrow CPU fixture, not a CUDA, Metal,
full-corpus quality, performance or numerical-tolerance qualification.
