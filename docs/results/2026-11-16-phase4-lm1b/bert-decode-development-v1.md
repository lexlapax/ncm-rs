# LM1B BERT scorer decoding development check

## Initial development check

The pinned upstream FLM scorer calls `bert.batch_decode(rows)` with default
options and retains special tokens. Two independent captures from development
seeds 11–15 and eight synthetic cases produced identical SHA-256
`fee7a02040abf97b7a64991d31e616133ace1ea37e574cdb018592d4ee19d6ee`.
The capture verifies the existing five input hashes, BERT manifest and source
files, and tokenizer package versions. It covers 1,280 previously inspected
128-token development rows; no official samples are included.

The BERT vocabulary has 30,522 IDs while LM1B's model vocabulary has 30,528.
None of the captured development rows contains one of the six additional
IDs. Pinned Python omits those six IDs during decoding; the generic Rust
tokenizer deliberately rejects them as outside its vocabulary. The probe
checks that rejection and, solely for this comparison, explicitly filters the
six model-valid IDs before decoding. All 1,288 texts then match exactly,
including four synthetic rows containing those IDs and synthetic special-token
cases. IDs at or above 30,528 are rejected by the probe. An eventual LM1B FLM
scoring adapter must implement and test this bounded upstream behavior; no
production decoder behavior was changed here. This is not model parity or a
quality gate.

Commands, run from the repository root:

```sh
.reference/tokenizers-venv/bin/python tools/reference/lm1b_bert_decode_reference.py --output reports/phase4-lm1b-bert-decode-reference-v1.json
.reference/tokenizers-venv/bin/python tools/reference/lm1b_bert_decode_reference.py --output reports/phase4-lm1b-bert-decode-reference-v2.json
RUSTC_WRAPPER= cargo run --locked -p ncm-tokenizers --release --example lm1b_bert_decode_probe -- artifacts/canonical/tokenizers-qualified/bert reports/phase4-lm1b-bert-decode-reference-v1.json
RUSTC_WRAPPER= ./tools/check.sh > reports/phase4-lm1b-bert-decode-check-v2.log 2>&1
```

The first standard check stopped on formatting; the example was formatted and
the second check passed afterward, including 518 Python tests. Captures and check logs are ignored local
artifacts, not model weights or tracked data.

## Rust FLM decoder and CLI text output

`TextTokenizer::decode_lm1b_flm` uses the authenticated BERT asset
and retains special tokens. It validates the LM1B model ID range
before passing IDs to the pinned WordPiece decoder. WordPiece omits
IDs 30522 through 30527 because they have no BERT vocabulary entry.
IDs at or above 30528 and oversized requests return errors. The generic
decode entry point checks every ID against its tokenizer vocabulary.

The pinned Python reference repeats byte for byte. The Rust adapter
matches all 1,288 decoded strings, including all four padded-ID
synthetic rows and the special-token and empty cases. The probe also
checks that generic BERT decoding rejects rows containing padded IDs,
and that the LM1B adapter rejects 30528, `u32::MAX` and an over-limit
array of padded IDs.

The [experimental CLI](../../PHASE4-TEXT.md#experimental-lm1b-small-cuda-command)
uses the embedded pinned BERT asset when `--decode` is present. On the
RTX 3060 Ti, five requests at batches 1, 8, 16 and 32 produce 65 exact
text rows against Python's `bert.batch_decode`. Each request's raw
output matches its archived bytes. The `--decode` response includes
those same codes and IDs together with the texts. The CLI rejects a
duplicate decode option before opening the artifact.

The feature-enabled CLI unit tests and Clippy pass. The full Linux
host check passes, including all 533 report and data tests.

The reference, release executable hash, output receipts and control log
are in `reports/phase4-lm1b-bert-adapter-v1/`.

| Receipt | SHA-256 |
| --- | --- |
| `reference.json` | `fee7a02040abf97b7a64991d31e616133ace1ea37e574cdb018592d4ee19d6ee` |
| `decode-probe.log` | `07795fab7c62aa73627c72157bae5f25b72643d7a16f08baa31d60a3943db805` |
| `smoke.py` | `b9d94ba591743a253881cefaf8625837eafd4e05cc3dffbc3d23cc0ee419c9fa` |
| `smoke.json` | `bfae0969cb24e85fd3f938d81933a4eca6d6f1219e766bab3a6a29ff6be107ae` |
| Release CLI | `9cd76a7bc3d21b78f62faca9d75941ae7f81234509960132ddada8137d7028d7` |
| `host-check.log` | `45ddb05b5dc4b1c786ccbc2215b177e1c422bb8ccaa95abef795454883014f99` |

The comparison covers observed decoding and the added CLI option.
[LM1B-small quality qualification](small-heldout-cuda-v1.md) remains open.
