# Phase 5 text input: XSum and WMT14

2026-11-19. `ncm sample ARTIFACT DEVICE INPUT_JSON --text` now accepts a JSON array of strings for XSum and WMT14. It uses the artifact's authenticated tokenizer; XSum truncates to 512 IDs and pads with 50257, WMT14 truncates to 96 and pads with 151643. At this screen TinyStories and LM1B rejected `--text`; TinyStories support was added later (below). Correction: TinyStories accepts any single valid non-padding prefix token, not only a fixed start token; LM1B requires `[101]`. This is not a completed Phase 5 text-input gate.

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

## Synthetic truncation and padding boundaries

Seven synthetic strings per profile exercise empty input, a single token, long repeated text and
whitespace, Unicode and special-token spellings. The pinned Python tokenizers produced untruncated
lengths `0, 1, 1101, 1400, 10, 1, 7` (XSum) and `0, 1, 1101, 350, 6, 1, 1` (WMT14). The Rust
CLI encoder matched every padded/truncated ID row, including overlength cases. Empty or special
strings passing this encoder-only check are not necessarily accepted by the downstream model.
No GPU generation or quality conclusion follows from this encoder-only test. A separate
[CUDA overlength screen](cuda-text-truncation-v1.md) compares generation for one such input per profile.

The reference generator verifies the pinned tokenizer manifest/source files and package versions,
then rechecks source hashes after tokenization. Its SHA-256 is
`23e1bac178719fde166e7bd5d08f3607aaa21a3ad530a8fcd07411e0aa709bc1`. Three independent
captures per profile matched byte-for-byte; final XSum SHA-256 is
`2f7c8ed36cfd669a59ecf2fc5e13fe2fb584941c61c93e069a0825a7c5ff46f6` and WMT14 is
`03886e0864b60f9fc7992311f83b3e20511b9ce73274fef5f51624d41d960fb6`. The pinned XSum tokenizer
warned about the deliberately overlong 1,101-token input; no model was invoked by the generator.
The offline output preflight test, the artifact-bound Rust test below, and `tools/check.sh` passed.
Choose fresh output paths:

```sh
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_text_boundaries.py xsum reports/NEW-xsum-boundaries.json
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_text_boundaries.py wmt14 reports/NEW-wmt14-boundaries.json
setpriv --inh-caps=-all env \
  NCM_XSUM_ARTIFACT="$PWD/artifacts/canonical/xsum-inference-rope-v2-final" \
  NCM_WMT14_ARTIFACT="$PWD/artifacts/canonical/wmt14-inference-v1-final" \
  NCM_XSUM_BOUNDARY_REFERENCE="$PWD/reports/NEW-xsum-boundaries.json" \
  NCM_WMT14_BOUNDARY_REFERENCE="$PWD/reports/NEW-wmt14-boundaries.json" \
  cargo test --locked -p ncm-cli --bin ncm synthetic_text_boundaries_match_pinned_python -- --ignored
```

## End-to-end CUDA screen

For the first corpus row of each profile, four fresh processes compared equivalent ID and `--text` input, both without and with `--decode`. Stdout matched byte-for-byte in each pair and matched the corresponding earlier retained baseline output. The common hashes are XSum `87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca` (IDs) and `6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328` (decoded), WMT14 `17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a` and `fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32`. The release binary built with `--features cuda-cudnn` had SHA-256 `9a80df0c2087d7f39e7b05971ae515799d69a6991971255a78299cd622b0d3df`. Build and runtime environment are as in [the earlier screen](cuda-routing-screen-v1.md). This proves the tested CUDA requests only; Metal and broader end-to-end cases are pending.

## Full-corpus CUDA text/ID generation retest

On 2026-11-20, `ncm sample --text` was run on **all 11,334 XSum** and **3,003 WMT14** retained corpus inputs with the candidate CUDA CLI. Its codes and IDs matched the previously retained equivalent ID-input CUDA captures for every row: 356 XSum batches (two interleaved shards, batch 32 and partial tails) and 751 WMT14 batches (batch four and a three-row tail). The [prior XSum CLI corpus comparison](cuda-xsum-quality-retest-v1.md) bound those XSum captures to the same candidate ID-input CLI; the [WMT14 comparison](cuda-wmt14-quality-retest-v1.md) separately bound all candidate CLI ID outputs to the same retained fast-backend capture. These are previously observed corpora, not new blind quality experiments. The known XSum intermediate-code tie and WMT14 CFG=1 failure remain disclosed in those reports.

Text and padded ID references were hash-checked against the full-corpus encoder evidence above. XSum text is reconstructed from the converted IDs, not the unavailable original articles. The runner verified each XSum capture against the prior CLI summary's per-batch hashes and each WMT14 row against the pinned full capture; it checked binary, artifact manifests, references and retained captures before and after. An independent local replay re-read every saved text input and output, verified row coverage, output hashes and all codes/IDs against the retained captures. The CUDA run did not use `--decode`; retained-output decoder parity is [separately covered](cuda-decode-screen-v1.md). It does not establish Metal validation, performance or a complete release gate. The desktop GPU was shared.

The candidate binary SHA-256 was `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`; runner SHA-256 `de150b7ebc4dff13cb9b6300c35c9666dbf83d90865a284ff65ee48f0b2dd025`. Ignored summary SHA-256s are XSum `02d5a159eacf50fc175c7df3f05c748b69da65f4d018cbb1c4c1bb28960b6a32` and WMT14 `07c2f5f13da157b6153b03d2eb10cd23e879eda5af9a343a40f54d2a3625601d`; raw input/output batches and summaries remain locally under `reports/phase5-{xsum,wmt14}-text-corpus-cuda-v1/`, not durably published. Smoke runs, batch-plan/negative-control unit tests and `setpriv --inh-caps=-all tools/check.sh` passed. No production code changed.

The full run used the original runner and checkout path. The retained XSum summary keys batch hashes
by absolute path; the runner was subsequently generalized to index unique shard/ordinal suffixes
without trusting that prefix. It requires all 356 identities, rejects duplicate, missing, malformed
or traversal entries, and still verifies the original summary and every capture hash. Offline
negative controls and `setpriv --inh-caps=-all tools/check.sh` passed. A two-batch CUDA smoke with
`ROOT` redirected to a separate local directory containing symlinks to the retained capture
directories produced outputs byte-identical to the first two full-run batches. This is a relocated
lookup smoke, not a fresh clone or a repeat of the full corpus. The updated runner SHA-256 is
`0a7b4ecbf49f21d0547df15f5556e35657565840f7446763b6a6c4ca5aacda41`; the ignored smoke summary
SHA-256 is `76cd1d96898edbdcb829adc3b5644f8c37b7ec9641651ccff7cf20362352b33a`.
The command below uses the updated runner and still requires the retained inputs and captures.

```sh
for profile in xsum wmt14; do
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
    python3 tools/bench/phase5_text_corpus_generation.py --profile "$profile" \
    --candidate target/release/ncm --reference "reports/phase5-text-corpus-v1/$profile.json" \
    --artifacts artifacts/canonical --output "reports/NEW-$profile-text-corpus"
done
```

## Selected multi-batch CUDA text/ID generations

On 2026-11-20, the pinned old alias with IDs, current unified command with IDs and `--decode`, and current unified command with `--text --decode` were run for each of XSum and WMT14. The selected corpus indices were `0`, `1234..1237`, and the final 32 rows: `11302..11333` for XSum and `2971..3002` for WMT14. For all six batches, old and current codes/IDs were equal, and the two decoded current stdout streams were byte-identical. The `--text` strings and corresponding padded IDs come from the independently pinned corpus references above. XSum strings remain reconstructions from converted IDs, not unavailable original raw articles. This screen covers **37 rows per profile**, not corpus-wide generation, quality, performance or Metal.

The baseline CLI SHA-256 was `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`, candidate `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`; the runner checked both binaries, reference and artifact manifest hashes before and after. The runner SHA-256 was `742fd6e7c591c980c79fc8b5349f313d6e7379863da5c730332d713f6b7eca35`. The ignored result JSON SHA-256 is `cd92ea18ad917d5fe694e0cce8393b167d66e8dd2c0801d084600a3d0005984e`; raw inputs and outputs remain under `reports/phase5-text-selected-cuda-v1/`. The desktop GPU was shared. Offline negative controls and `setpriv --inh-caps=-all tools/check.sh` passed; no production code changed.

Reproduce with the corpus references generated as above and the pinned artifacts and binaries:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
  LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
  python3 tools/bench/phase5_text_generation_screen.py \
  --baseline reports/phase5-cli-equivalence/baseline-ncm --candidate target/release/ncm \
  --references reports/phase5-text-corpus-v1 --artifacts artifacts/canonical \
  --output reports/NEW-text-selected-cuda
```

## TinyStories prefix correction

The current TinyStories teacher accepts one prefix ID per row in `0..8192`, excluding pad ID 1; it is **not** restricted to start ID 0. On CUDA, seed 42 and prefix `[[2]]` produced byte-identical stdout through the pre-change binary, candidate alias and candidate unified command (SHA-256 `efe7256404004b6551c25868fabebb2c2d06f12841662cb907473db34d11d03c`). It differed from the prior `[[0]]` output at 639 of 641 codes. This is a two-input development screen, not a claim that arbitrary multi-token prefixes are supported or qualified.

## TinyStories text input

`--text` now encodes using the artifact tokenizer and requires exactly one non-padding token per input string; it does not truncate multi-token input. The pinned upstream tokenizer encodes `a`, `hello` and `<|endoftext|>` as `[99]`, `[5663]` and `[0]`; Rust's artifact-bound tokenizer agrees. On CUDA with seed 42, the three-row text request produced byte-identical stdout to the equivalent IDs via the baseline, alias and unified commands (SHA-256 `5657f32f642395268128b19df66e283ab5644ae59bc4869a0e542c467b5d2f63`). With `--decode`, text and ID inputs also matched (SHA-256 `ba9da3f1183b3e7a025693af4d3218ee431d61fc0af9bc87e27b4cbe4d364cfd`), and the non-text fields match the baseline. Empty, `<|pad|>`, `hello world` and `😀` inputs were rejected explicitly. Candidate release binary SHA-256 `cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`. The first screening harness accidentally overwrote its own text input with an output; that failed run was discarded and the clean rerun used disjoint filenames. Raw captures are in ignored `reports/phase5-tiny-text-v2/`. Metal and performance checks are pending.

### TinyStories vocabulary differential

The pinned upstream TinyStories byte-BPE decoded each of its 8,192 vocabulary entries to text,
then encoded that text; seven additional boundary strings were appended. Rust's actual
artifact-bound text-input path matched upstream **admission and ID** for all 8,199 rows: 8,064
single non-padding rows accepted with identical IDs, and 135 zero-token, padding or multi-token
rows rejected. For rejected rows the test checks rejection, not equality of the full encoded
sequence. Some decoded byte tokens contain Unicode replacement characters or re-encode to
multiple IDs, so this is not a claim that every vocabulary ID round-trips as text. It is not a
GPU generation or quality test.

The reference generator checks pinned upstream source hashes, clean Git revision, tokenizer
payload/manifest and locked dependency versions. Two captures were byte-identical, SHA-256
`b693f72b52e4803b8604774f63f3edc5eaed1b609136d2d0218c76485068add3` (ignored locally);
the generator SHA-256 is `22644477c89aa619ab096e2f8a57dba8b11c2d63717fc6867153a33f06f55be3`.
Run with a fresh reference output:

```sh
setpriv --inh-caps=-all .reference/tokenizers-venv/bin/python tools/reference/phase5_tinystories_text_reference.py \
  --upstream .reference/upstream-ncm --tokenizer-asset artifacts/canonical/tokenizers-qualified/tinystories \
  --output reports/NEW-tinystories-text-reference.json
setpriv --inh-caps=-all env NCM_TINYSTORIES_ARTIFACT="$PWD/artifacts/canonical/tinystories-phase4-v1" \
  NCM_TINYSTORIES_TEXT_REFERENCE="$PWD/reports/NEW-tinystories-text-reference.json" \
  cargo test --locked -p ncm-cli --bin ncm tinystories_text_matches_upstream_vocabulary_reference -- --ignored
```

The ignored Rust test verifies the reference hash and all rows. Two reference preflight negative
controls and the CLI feature tests/Clippy pass. The first `tools/check.sh --cuda` failed an
unrelated `test_default_budget_handoff_preserves_deadline_and_all_eighteen_real_observations`
with `capture_error:ObservationMismatch` / unexpected live executable identity; a focused retry
and two later full CUDA checks passed. The cause is not established; the failed log remains at
`/tmp/ncm-phase5-tiny-vocab-cuda.log` for this session.
