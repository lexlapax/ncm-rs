# Phase 2 tokenizer assets and Linux parity

Scope: Sudoku regression plus TinyStories byte BPE and the GPT-2/Qwen/BERT wrappers
referenced by released profiles. This is **not full text-model inference**, chat
templating, tokenizer training or a generic importer. Performance measurements and
the Apple pass remain pending the corresponding platform gates.

## Exact source and runtime identity

Original NCM remains `c70274273f489abc73832b2b98584a8483b6b203`.
`tools/reference/tokenizer-sources.lock.json` records every file's size, SHA-256,
Git blob identity and repository revision; initial fetch verifies Git blob or LFS
identity before conversion. Repeat fetching accepts `--lock` to require those exact
records. No text-model weights were downloaded.

| Profile | Repository | Immutable revision | License |
| --- | --- | --- | --- |
| TinyStories | hazyresearch/ncm-checkpoints | d342ce8aaea58830eef3b8ad473664673e41f692 | MIT |
| GPT-2 / XSum | openai-community/gpt2 | 607a30d783dfa663caf39e06633721c8d4cfcd7e | MIT |
| Qwen3 / WMT14 | Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca | Apache-2.0 |
| BERT / LM1B | google-bert/bert-base-uncased | 86b5e0934494bd15c9632b12f734a8a67f723594 | Apache-2.0 |

Canonical bundles retain the exact license notice. GPT-2's notice comes from
OpenAI GPT-2 commit `9b63575ef42771a015060c964af2c3da4cf7c8ab`; TinyStories retains
the pinned NCM license, and Qwen/BERT retain their source LICENSE files. The bundle
loader checks these notices as well as tokenizer data. Source/model-card files
remain in the ignored source cache; trained weights/data are not in Git.

The new isolated Python reference environment pins Transformers **4.57.6**,
Tokenizers **0.22.2**, and regex **2025.9.18**. The latter deliberately uses Unicode
**16.0**, matching the locked Rust regex-syntax tables used for TinyStories rather
than silently comparing different Unicode classifications. NCM leaves these
reference dependencies unpinned; these are explicit new tokenizer oracle pins,
not changes to any established model golden or Torch environment. Full dependency
lock: `tools/reference/requirements-tokenizers.lock`.

Rust uses Tokenizers **=0.22.2** with only its Oniguruma feature (no Hub/HTTP or
remote-code support) and the already-locked fancy-regex **=0.17.0** for byte BPE.
Candle's separate existing Tokenizers 0.23.2 dependency is unchanged; the two
versions are deliberate, not an implicit reference upgrade.

## Conversion and runtime checks

- TinyStories vocabulary: 8,192 unique byte strings; 7,934 ordered merges. A bounded
  protocol-four **data-opcode interpreter** accepts only the observed scalar/
  container opcodes. It does not call `pickle.load`, an Unpickler, globals, reduce
  functions or object constructors. Framing, input/operation/memo/stack bounds,
  duplicate keys, operand types and ranked vocabulary consistency are checked.
- Bytes remain hex in canonical JSON; no lossy text conversion of the vocabulary.
  All 256 byte IDs, EOS 0 and PAD 1 are fixed. Native merging applies every occurrence
  of the lowest-ranked available pair, preserving the upstream algorithm.
- HF assets are loaded offline by the pinned upstream wrapper with
  `trust_remote_code=False`, `local_files_only=True` and the required fast tokenizer.
  The resulting native tokenizer JSON retains normalization, postprocessing, added
  tokens and decoder options. No slow or remote-code fallback is allowed.
- `TextTokenizer::load` accepts only the four reviewed canonical manifest/payload
  hashes in `tools/reference/tokenizers-v1.lock.json`. Unknown or changed metadata,
  payloads, licenses and symlink leaves/roots fail. It is not a generalized importer.
  Parent directories must be trusted and free of concurrent hostile mutation.
- Bounds: manifest/license 64 KiB, payload 32 MiB, text 1 MiB, ID count 1 MiB;
  byte-BPE decoded bytes 16 MiB. These are byte/request limits, not peak-RSS claims.
  Invalid token IDs fail explicitly, including when skipping special tokens.

| Profile | Base / total vocabulary | EOS / PAD | Extra HF cleanup | Encode cases | Decode inputs (each tested with skip=false/true) |
| --- | ---: | ---: | --- | ---: | ---: |
| TinyStories | 8192 / 8192 | 0 / 1 | false | 89 | 351 |
| GPT-2 | 50257 / 50258 | 50256 / 50257 | false | 89 | 349 |
| Qwen3 | 151643 / 151669 | 151645 / 151643 | false | 89 | 361 |
| BERT | 30522 / 30522 | 101 / 0 | false | 89 | 96 |

**All 356 encode and 2,314 decode comparisons pass exactly.** Cases cover empty
input, controls/whitespace, contractions, combining characters, multiple scripts,
Unicode-16 letters, emoji/ZWJ, special tokens, BERT postprocessing/normalization,
all byte-valued decoder tokens for BPE profiles, invalid UTF-8 fragments, and 64
seeded mixed-string cases per profile. HF extra cleanup is false in this pinned
wrapper; BERT's native WordPiece decoder still retains its own cleanup behavior.
A low-level extra-cleanup helper is independently tested against the Transformers
replacement order, not silently applied to these profiles.

The real-asset probe also rejects four mutations per profile (manifest, payload,
license and payload symlink). Six new base Rust tests cover byte merging, malformed
schemas/hex, Unicode, cleanup, bounds and symlink-root/leaf rejection; the three
existing Sudoku tests remain unchanged. Five new offline tests reject executable
pickle opcodes, wrong schemas, duplicate keys, malformed frames and trailing data.

Two fresh conversions produce identical fixtures and canonical locks. Adding license
packaging changed metadata only; all previously captured encode/decode values were
explicitly compared and stayed unchanged. Fixture SHA-256:
`5c937a5e5aef961d9361b7f5788929681b24e55c223886d11b5928169db23256`.
`raw/native.json` includes source/asset fingerprints and exact counts. The release
probe passes with a cleared environment and no Python on PATH; its Linux dependency
list contains libc/libgcc_s and the loader, not Python.

## Reproduce

```sh
python3 -m venv .reference/tokenizers-venv
.reference/tokenizers-venv/bin/python -m pip install -r tools/reference/requirements-tokenizers.lock
python3 tools/reference/fetch_tokenizers.py artifacts/upstream/tokenizers-v2 \
  --lock tools/reference/tokenizer-sources.lock.json
.reference/tokenizers-venv/bin/python tools/reference/convert_tokenizers.py \
  artifacts/upstream/tokenizers-v2 .reference/upstream-ncm artifacts/canonical/tokenizers-qualified
cargo run --release -p ncm-tokenizers --example asset_probe -- \
  artifacts/canonical/tokenizers-qualified tests/fixtures/tokenizers-v1.json
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
.reference/venv/bin/python -m unittest discover -s tools/reference -p 'test_*.py' -v
```

Use fresh fetch/conversion paths. The full optional reference test command uses the
existing Torch environment; the tokenizer converter itself needs no Torch. On
Linux, full checks pass: **30 base Rust tests**, the real-CUDA codec suite, all 18
unchanged operator cases, seven report-tool tests, fmt/lint/docs/syntax; all **18**
offline reference tests pass. Log: `.reference/phase2-tokenizer-check.log`.

No latency/throughput/memory comparison is claimed here; those measurements remain
required before the Linux Phase 2 gate. No Phase 2 Apple job has been started.
