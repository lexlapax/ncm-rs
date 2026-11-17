# LM1B BERT scorer decoding development check

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
