# Canonical artifact contract proposal (v1)

To implement in Phase 1; this document is not a working format/parser yet.
Canonical runtime files must be portable across both adapters and preserve the
pinned Python reference without executable serialization.

```text
<artifact-root>/
  manifest.json
  config.json
  codec.safetensors
  generator.safetensors
  inference.safetensors
  tokenizer.json                  # HF profile, including added tokens/postprocessors
  byte-bpe.json                    # TinyStories alternative; byte-safe encoding
  provenance.json
```

Large models may shard safetensors with a deterministic tensor index. Trainable
weights, inference statistics and training/EMA state must be distinguished. A
future training bundle extends the schema rather than repurposing inference files.

## Manifest/provenance requirements

- Schema version, model/task identity and explicit compatibility profile.
- Upstream repository + commit, checkpoint repository + immutable revision,
  per-input SHA-256 and output SHA-256, converter Git commit/version.
- Resolved architecture, level schedule, all defaults, dtype policy, tensor names,
  shapes, byte counts, canonical layouts, aliases/tied-weight ownership.
- Tokenizer repository/revision, special token values/IDs, added tokens,
  normalizer/pretokenizer/postprocessor/decoder and cleanup policy.
- Generation defaults: level-zero source and unigram, token allowances,
  temperature/top-k/top-p/CFG, penalties and EOS handling.
- Code/weights/tokenizer/data licenses and attribution separately.
- Python/Torch/export dependencies and relevant device/math settings; documented
  reference patches. No environment secrets or machine-specific absolute paths.

Store weights in documented canonical (upstream-compatible) layouts. Adapters
transform once on load to backend layouts; never silently rewrite the shared
artifact for one device. Recompute nonpersistent derived values (RoPE, tied-head
codebook norms/scales) under explicitly tested rules or record their exact reference
forms. Generator/codec association is mandatory: a generator alone is insufficient.

## Conversion and loading security

Use trusted, pinned inputs and `torch.load(..., weights_only=True)` wherever
supported. Never automatically retry with unrestricted pickle. Legacy byte-BPE and
evaluation pickle conversion requires a separate explicit trusted-input operation
inside a restricted offline environment, without secrets. Production Rust accepts
only checked data formats, not pickle, Python config execution or remote model code.

Validate hashes, file sizes/shape products, allowed dtypes, duplicates, missing and
unexpected tensors, bounds and model/profile agreement before uploading to GPU.
Reject absolute/escaping paths, unsafe symlinks and unknown schema versions. An
explicit path resolver must flatten upstream `../tokenizer.pt` into a safe declared
artifact relationship rather than allow arbitrary manifest traversal.

Maintain an allowlist of intentionally omitted non-inference buffers; no blanket
`strict=false` equivalent. Report ignored training settings distinctly from
unsupported inference settings. Normalize tied/shared weights without losing
aliases or introducing independently mutable copies.

Memory mapping, if used, must retain file ownership for the entire tensor lifetime
and document mutation/truncation hazards; a safe copied-load path comes first.
All downloads are explicit user actions with revision/checksum validation and a
cache-size budget. Never download the entire multi-GB repository in ordinary CI.

## Dataset and result formats

Convert evaluation inputs to bounded JSONL (text/IDs and stable sample IDs) or
another reviewed non-executable columnar format. Keep shard/sample/seed identity
explicit to avoid output ordering mistakes. Record raw IDs as well as decoded text
for FLM and entropy protocols. Generated text/datasets may have redistribution
limits: publish aggregate metrics and checksums when raw release is inappropriate.

See [UPSTREAM.md](UPSTREAM.md) for schema evolution and [REPORTING.md](REPORTING.md)
for immutable result/provenance bundles.
