# WMT14 bounded CUDA capture preparation

The experimental Python and Rust capture tools accept a bounded contiguous
range within the 3,003 verified prefixes, using batches of four and a smaller
last batch when required. They retain generated codes and raw IDs; scoring and
performance are separate. The upstream tool checks the clean pinned source,
checkpoint and prefix identities before and after execution. The Rust tool
checks the artifact manifest and prefix identity, but does not perform source
postflight. Both create a new final output only after processing the range;
neither supports resuming a partial run. Preserve logs on failure.

The first 128 rows are designated development data, including the previously
inspected 32 rows. Use development baseline variation and metric behavior to
choose and record justified non-inferiority criteria before comparing held-out
quality. Rows 128–3002 remain held out for that decision. The full official
3,003-row BLEU, lowercase BLEU, chrF and length-ratio evaluation, and separate
base/CFG runs, remain required. The upstream scorer dependency is not yet
pinned in the local reference environment.

The following smoke used 35 rows, exercising eight four-row batches and a
three-row tail; a separate run used start 32 and count 3:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_cuda_capture.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-python-cuda-capture-smoke-v1.json --start 0 --count 35
setpriv --inh-caps=-all env RUSTC_WRAPPER= CUDA_COMPUTE_CAP=86 cargo build --release -p ncm-backend-cuda --features wmt14-experimental --example wmt14_capture
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_capture artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json reports/phase4-wmt14-rust-cuda-capture-smoke-v1.json 0 35
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_cuda_capture.py reports/phase4-wmt14-python-cuda-capture-smoke-v1.json reports/phase4-wmt14-rust-cuda-capture-smoke-v1.json reports/phase4-wmt14-cuda-capture-smoke-comparison-v1.json --reference-sha256 cea59ae16487d418e06013696bc54fc4985ef34cd243cd17f4920b5b93dc61f3 --candidate-sha256 e2fc30867e5d73a3ffb09e8f1f95f508d43786992da3852a31ab0a12c2180958
```

Both ranges matched all codes and raw IDs exactly. The 35-row comparison
SHA-256 is `38f074de20d4b8572397f0674214db3403c78e452941c5c55e8626bd01e934c6`.
The Rust capture binary SHA-256 is
`fd2c725f61cb41bc478ebd6a8b1a96256a6ed0b4c6bd5b4f9fd3886a57b8007f`;
the Python capture script SHA-256 is
`643a73203d0a65cc454548d5020482652a59d7774dc54a5c4ddb5e5a148f6f67`.
The comparator range/row regression, invalid-range rejection and feature
Clippy pass. These are development smokes, not quality, performance or Metal
qualification. Desktop GPU processes were present.
