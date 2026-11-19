# Phase 5 text input: XSum and WMT14

2026-11-19. `ncm sample ARTIFACT DEVICE INPUT_JSON --text` now accepts a JSON array of strings for XSum and WMT14. It uses the artifact's authenticated tokenizer; XSum truncates to 512 IDs and pads with 50257, WMT14 truncates to 96 and pads with 151643. TinyStories and LM1B still explicitly reject `--text`. Correction: TinyStories accepts any single valid non-padding prefix token, not only a fixed start token; LM1B requires `[101]`. This is not a completed Phase 5 text-input gate.

## Corpus-exact encoding

The actual Rust encoder used by the CLI matched **every** pinned reference row: 11,334 XSum rows and 3,003 WMT14 rows. The XSum converted corpus contains IDs, not original article strings: the reference reconstructs text by decoding the first 512 IDs with the pinned GPT-2 tokenizer, stripping trailing padding and disabling cleanup. The Rust encoder reproduces every original 512-ID prefix. This does not establish equivalence to unavailable original raw articles. WMT14 reference strings are the actual converted source corpus; expected IDs come from the pinned Qwen3 tokenizer.

The committed generator verifies input SHA-256, tokenizer source files and package versions, and writes a new file without overwriting. Reference generation was replayed byte-identically; SHA-256 is `f51a1232bcd47169cd970edd61b597b5aebef12f0f202e6d51f41be6642faa2f` (XSum) and `4c1d6b68e89b46800219e9099116b1a50d51b2c96a51bdc8abe6dec6a98add9e` (WMT14). Reproduce from the repository root, using fresh output paths:

```sh
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_text_inputs.py xsum reports/NEW-xsum-text.json
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_text_inputs.py wmt14 reports/NEW-wmt14-text.json
setpriv --inh-caps=-all env \
  NCM_XSUM_ARTIFACT="$PWD/artifacts/canonical/xsum-inference-rope-v2-final" \
  NCM_WMT14_ARTIFACT="$PWD/artifacts/canonical/wmt14-inference-v1-final" \
  NCM_XSUM_TEXT_REFERENCE="$PWD/reports/NEW-xsum-text.json" \
  NCM_WMT14_TEXT_REFERENCE="$PWD/reports/NEW-wmt14-text.json" \
  cargo test --locked -p ncm-cli --bin ncm corpus_text_encoding_matches_every_reference_row -- --ignored --nocapture
```

This explicit ignored test passed on Linux. It checks both row counts and compares every encoded padded row, in batches of at most 128. The test reads the complete validated artifacts; it does not itself check the external reference hashes, so use the generator above.

## End-to-end CUDA screen

For the first corpus row of each profile, four fresh processes compared equivalent ID and `--text` input, both without and with `--decode`. Stdout matched byte-for-byte in each pair and matched the corresponding earlier retained baseline output. The common hashes are XSum `87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca` (IDs) and `6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328` (decoded), WMT14 `17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a` and `fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32`. The release binary built with `--features cuda-cudnn` had SHA-256 `9a80df0c2087d7f39e7b05971ae515799d69a6991971255a78299cd622b0d3df`. Build and runtime environment are as in [the earlier screen](cuda-routing-screen-v1.md). This proves the tested CUDA requests only; Metal and broader end-to-end cases are pending.

## TinyStories prefix correction

The current TinyStories teacher accepts one prefix ID per row in `0..8192`, excluding pad ID 1; it is **not** restricted to start ID 0. On CUDA, seed 42 and prefix `[[2]]` produced byte-identical stdout through the pre-change binary, candidate alias and candidate unified command (SHA-256 `efe7256404004b6551c25868fabebb2c2d06f12841662cb907473db34d11d03c`). It differed from the prior `[[0]]` output at 639 of 641 codes. This is a two-input development screen, not a claim that arbitrary multi-token prefixes are supported or qualified. `--text` for TinyStories is still not implemented.
