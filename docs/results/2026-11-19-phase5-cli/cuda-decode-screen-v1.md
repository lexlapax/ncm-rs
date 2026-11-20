# Phase 5 text decode: CUDA screen

2026-11-19. On the real CUDA device, the unified command with `--decode` was run on the fixed requests in [the routing screen](cuda-routing-screen-v1.md). For each request, removing `texts` from the result reproduced the baseline codes, IDs and other fields exactly. The text for each of the six text profiles matched its pinned evaluator decoding on the generated IDs: the upstream TinyStories `BPETokenizer`, Transformers GPT-2 and Qwen3 for XSum and WMT14, and Transformers BERT for all three LM1B sizes. This is one output row per profile, **not** corpus-wide or Metal evidence.

| Profile | Decoded CLI stdout SHA-256 |
| --- | --- |
| TinyStories | `5dcc981e5e8bfa63e1d9c17f097e2213a1449be0ec9ac5d16618e7f96054cc4f` |
| XSum | `6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328` |
| WMT14 | `fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32` |
| LM1B small | `cd87171d7f866ad664cedf252c8eee86e857d5c66a00855468da73e009a423c6` |
| LM1B medium | `c2995c6aaaa74f764f6602d6105aa0f51c121a2f752371d5cd6653e36f5a143e` |
| LM1B large | `398b182c08dea307d3518c2382fd958d9e5b9bddf3870da33996e455e3bb44d1` |

The candidate binary SHA-256 was `12eb0b9fc45518cf4d01cab73b7316eec147f78159552466ec1f7a9153d37074`, built `--release --locked -p ncm-cli --features cuda-cudnn` with the build environment from the routing screen. The retained inputs, baseline and decoded JSON are in ignored `reports/phase5-cli-equivalence/`. Run with `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and the reference environment's cuDNN `LD_LIBRARY_PATH`.

For TinyStories, XSum and WMT14, the reference stops at the first EOS, removes padding, and decodes with special tokens skipped. LM1B uses the existing FLM behavior: decode the full row with BERT specials retained. GPT-2 and Qwen3 reference decoders return an empty string for their model-only padded-vocabulary IDs; the CLI filters those IDs within the model vocabulary before decoding. Unit tests exercise EOS, padding, model-only IDs and out-of-range rejection. The reference comparison used the pinned assets under `artifacts/upstream/tokenizers-v2/` and `.reference/upstream-ncm`, with `.reference/tokenizers-venv`.

The original screen left text encoding, LM1B rotary setup, broader decode checks, Metal validation
and performance evidence pending. Later work is recorded in the [current context map](../../../agent-context-map.md);
no release gate is claimed by this screen.

## WMT14 retained CUDA corpus decode

On 2026-11-20, the artifact-bound `Decoder::add_texts` used by `ncm sample --decode` matched the pinned evaluator's decoded hypotheses on all **3,003** retained WMT14 CFG=0 CUDA output rows from the [quality retest](cuda-wmt14-quality-retest-v1.md). The ignored regression test checks the hashes and linkage of the capture and evaluator output, every sequential row index, 64 IDs per row, and all decoded strings. The same IDs were independently matched to the unified CLI in the quality retest; this host-side test does not run the CLI or a GPU, test Metal, or cover other text profiles. No runtime code changed.

The retained `rust.json` SHA-256 is `d26b8a3b16a4ad07565f52d3e61f22b30e2512ea10348e674a24e9ed52fecc79`; `rust-text.json` is `0daf73dd5844b1dd832a9b9192e231d149c1be815b40cb833f55f4e81235c624` and names the capture hash. Both are ignored local files under `reports/phase5-wmt14-fast-cuda-full-v1/`; their generation and pinned reference are documented in the linked retest. A modified reference was rejected by the hash check as a negative control. The test and `setpriv --inh-caps=-all tools/check.sh` passed.

```sh
setpriv --inh-caps=-all env \
  NCM_WMT14_ARTIFACT="$PWD/artifacts/canonical/wmt14-inference-v1-final" \
  NCM_WMT14_CAPTURE="$PWD/reports/phase5-wmt14-fast-cuda-full-v1/rust.json" \
  NCM_WMT14_DECODED_CAPTURE="$PWD/reports/phase5-wmt14-fast-cuda-full-v1/rust-text.json" \
  cargo test --locked -p ncm-cli --bin ncm wmt14_retained_corpus_decode_matches_pinned_evaluator -- --ignored --nocapture
```

## TinyStories single-token vocabulary decode

The artifact-bound CLI decoder was checked on all 8,192 single-token IDs against the pinned
upstream TinyStories vocabulary reference generated for the [text-input differential](cuda-text-input-v1.md#tinystories-vocabulary-differential).
For IDs 2–8191, decoded strings matched upstream; EOS 0 and padding 1 produced empty output under
the evaluator policy. The ignored Rust test verifies the reference SHA-256 and checks every row.
This covers single-token decoding, not arbitrary multi-token sequences, generated corpus output,
GPU inference or Metal. It passed with the retained qualified artifact and reference. To reproduce,
first generate a fresh reference as described in the linked screen, then run:

```sh
setpriv --inh-caps=-all env \
  NCM_TINYSTORIES_ARTIFACT="$PWD/artifacts/canonical/tinystories-phase4-v1" \
  NCM_TINYSTORIES_TEXT_REFERENCE="$PWD/reports/NEW-tinystories-text-reference.json" \
  cargo test --locked -p ncm-cli --bin ncm tinystories_single_token_decode_matches_upstream_vocabulary -- --ignored
```

## Synthetic TinyStories multi-token decode

The artifact-bound CLI decoder matched the pinned upstream `BPETokenizer` on 137 synthetic rows:
nine empty, EOS/padding and split UTF-8 edge cases, plus 128 fixed-seed random 128-token rows.
The Python reference applies the evaluator's first-EOS and padding policy. The cases include an
emoji reconstructed from four separately tokenized bytes and incomplete byte sequences. The
ignored Rust test checks the exact reference hash and all decoded strings. This is host-side
synthetic coverage, not generated-corpus, GPU inference or Metal qualification.

The generator reuses the upstream revision, clean-checkout, source-hash, package-version and
TinyStories tokenizer preflight used by the single-token reference, and rechecks inputs after
capture. Generator SHA-256 is `84224a51cdc4101d7aeee6e7595a0b66ce01335f79705ca5688c842bfea50a86`;
two captures matched byte-for-byte at SHA-256
`0c2361bece994e027a1db7a9e93b856cf27cf8aff9c1b1194d39f979623f737b`. The output preflight and
artifact-bound test passed. Use a fresh reference path:

```sh
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_tinystories_decode_reference.py \
  --upstream .reference/upstream-ncm --tokenizer-asset artifacts/canonical/tokenizers-qualified/tinystories \
  --output reports/NEW-tinystories-decode-reference.json
setpriv --inh-caps=-all env \
  NCM_TINYSTORIES_ARTIFACT="$PWD/artifacts/canonical/tinystories-phase4-v1" \
  NCM_TINYSTORIES_DECODE_REFERENCE="$PWD/reports/NEW-tinystories-decode-reference.json" \
  cargo test --locked -p ncm-cli --bin ncm tinystories_multi_token_decode_matches_upstream -- --ignored
```

## Synthetic XSum and WMT14 decode boundaries

A separate Linux host-side test compared the CLI's actual artifact-bound decoder with pinned
Transformers evaluators on 135 synthetic ID rows per profile: seven empty/EOS/padding/model-only
boundary rows and 128 fixed-seed random 32-ID rows. All 270 decoded strings matched. The Python
reference applies the evaluator's first-EOS and padding policy and uses the pinned tokenizer with
special tokens skipped; the Rust path exercises `Decoder::add_texts`. This is not model-generated
corpus output, GPU inference or a Metal qualification.

`tools/reference/phase5_decode_boundaries.py` SHA-256 is
`32ca9289857342cfa0dfc6f32bce93bfbcd99ec73f1eb73463b1d1c9e38fa768`. It verifies the pinned
source files, manifest and package versions before tokenization and rechecks input hashes after.
Two independent captures per profile were byte-identical; reference SHA-256s are
`ce931c2380ee57da00aa84ed818b005ed27bd5c81e3265766b075f126cde377c` (XSum) and
`16bbc0f63d8fe2f28813b12554375e4a54d251cee420ac2602c450878bd0dc9c` (WMT14). The output
preflight test, artifact-bound Rust test and `tools/check.sh` passed. Reproduce with fresh paths:

```sh
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_decode_boundaries.py xsum reports/NEW-xsum-decode-boundaries.json
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_decode_boundaries.py wmt14 reports/NEW-wmt14-decode-boundaries.json
setpriv --inh-caps=-all env \
  NCM_XSUM_ARTIFACT="$PWD/artifacts/canonical/xsum-inference-rope-v2-final" \
  NCM_WMT14_ARTIFACT="$PWD/artifacts/canonical/wmt14-inference-v1-final" \
  NCM_XSUM_DECODE_REFERENCE="$PWD/reports/NEW-xsum-decode-boundaries.json" \
  NCM_WMT14_DECODE_REFERENCE="$PWD/reports/NEW-wmt14-decode-boundaries.json" \
  cargo test --locked -p ncm-cli --bin ncm synthetic_decode_boundaries_match_pinned_python -- --ignored
```
