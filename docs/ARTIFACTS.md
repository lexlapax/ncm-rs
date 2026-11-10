# Canonical artifact contract proposal (v1)

The general contract below remains a proposal. A narrower Phase 1 inspection
profile is implemented (see the next section); it is not yet a runtime-model schema.
Canonical runtime files must be portable across both adapters and preserve the
pinned Python reference without executable serialization.

## Implemented inspection profile

`tools/reference/sudoku_artifact.py` exports only the pinned Sudoku-easy pair as
`sudoku-easy-inspection-v1`: `manifest.json`, `config.json`, `tokenizer.json`,
`provenance.json`, `codec.safetensors`, `generator.safetensors`, and
`training.safetensors`. All 238 source tensors retain names, layouts, shapes and
F32/I64 dtypes; 36 EMA/count/hit buffers are separated, not discarded. This pair
has no aliases or inference-statistic metadata. Unexpected sharing/metadata is
rejected; unrestricted pickle is never retried. Input byte sizes and SHA-256 are
pinned in the exporter. Outputs refuse preexisting destinations; the output parent
must be trusted and not concurrently mutated.

`ncm-io::Artifact::load` copies and validates declared file/tensor inventories and
hashes. Limits: 4 MiB manifest, 256 MiB per payload, 512 MiB summed payloads, tensor
rank 8. These are input-byte limits, **not peak-RSS guarantees**. It rejects unknown
manifest fields, duplicate manifest-map keys, malformed hashes, unsupported pins,
shape overflow, unexpected declared paths and nonregular/symlink leaf files/root.
It ignores undeclared files on disk. Parent path components and the directory must
be trusted, with no concurrent mutation; this is not an `openat`/`O_NOFOLLOW`
sandbox. It never executes Python or downloads anything.

Integrity is relative to the supplied manifest, not authentication of that manifest.
Config/tokenizer/provenance JSON is syntax-checked but not model-semantically
validated. A malicious replacement manifest can describe different tensors;
callers need a trusted manifest digest plus later model-profile checks. The profile
preserves upstream config rather than pretending all omitted defaults are resolved.
Nonpersistent buffers are inventoried by the pinned oracle, not runtime-constructed
by Rust yet. Do not use this inspection loader as a complete inference importer.

See [reproduction commands](../tools/reference/README.md) and the
[qualification report](results/2026-11-10-sudoku-artifacts/README.md).

## Proposed general runtime layout

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
