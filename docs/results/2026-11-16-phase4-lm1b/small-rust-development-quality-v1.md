# LM1B-small Rust development sampling and FLM scoring

The opt-in Rust CUDA sampler generated 256 samples at batch size eight for
each non-held-out seed 11–15, with CLS prefix 101, sampled level-zero unigram,
full later distributions and the pinned final penalties. It uses strict FP32
cuDNN plus the explicit CPU FP64 inverse-CDF sampler. Each batch has a distinct
request-local SplitMix64 seed `(development_seed << 32) | batch_index`; this
does not reproduce the PyTorch random stream. The capture tool rejects official
seeds and checks the artifact manifest and output bounds. Seed 11 reproduced
byte-for-byte with the final capture binary. Its capture SHA-256 is
`fa2bf52276c1dc228a69d46f8af61ae38206f7bad05fa36a2b0571f67282b7f3`.
All five capture and scorer identities are fixed in
`tools/reference/lm1b_rust_development_summary.py`.

The existing pinned FLM scorer matched the upstream scorer exactly on each
Rust capture. Seed 11's scorer report also repeated byte-for-byte. Scores use
BERT decoding with special tokens retained, GPT-2-large retokenization and the
pinned upstream token-validity expression. Per-seed results are:

| Seed | Python perplexity | Rust perplexity | Python entropy | Rust entropy |
| --- | ---: | ---: | ---: | ---: |
| 11 | 92.428020 | 94.855463 | 4.3097050 | 4.3112375 |
| 12 | 90.220961 | 94.143298 | 4.3099179 | 4.3063337 |
| 13 | 91.645407 | 91.790388 | 4.3100453 | 4.3126406 |
| 14 | 93.132655 | 90.775756 | 4.3135786 | 4.3128092 |
| 15 | 97.452219 | 91.805001 | 4.3144380 | 4.3148636 |

Rust minus Python seed-mean perplexity is -0.301871 and entropy is
+0.00003997 nats. Independent seed-resampling percentile intervals, 10,000
resamples with seed 42, are [-3.037569, 2.096688] for perplexity and
[-0.00318109, 0.00293752] for entropy. These are descriptive with only five
seeds. The checked summary repeated byte-for-byte, SHA-256
`5f1bb198734436ec390969fa064a5723b1ae944c929c73e77278facbf375e3f6`.

These are development results, **not** the frozen official-seed quality gate.
The separate controlled-draw comparison [failed](small-controlled-cuda-v1.md),
normal model loading is unsupported, and performance and Metal remain open.
Other desktop GPU processes were present, so this run is not a resource
benchmark. No official samples were generated or scored here.

Commands, with each capture and score written to a fresh path:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_development_capture -- artifacts/canonical/lm1b-small-inference-v1 11 reports/phase4-lm1b-small-rust-dev-seed-11-v3.json
setpriv --inh-caps=-all env LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" target/release/examples/lm1b_small_development_capture artifacts/canonical/lm1b-small-inference-v1 SEED reports/phase4-lm1b-small-rust-dev-seed-SEED-v1.json
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/.reference/tokenizers-venv/lib/python3.14/site-packages" .reference/venv/bin/python tools/reference/lm1b_quality_probe.py --upstream .reference/upstream-ncm --model artifacts/upstream/gpt2-large-phase4 --bert artifacts/upstream/tokenizers-v1/bert --development-capture CAPTURE --development-sha256 CAPTURE_SHA256 --output SCORE
.reference/venv/bin/python tools/reference/lm1b_rust_development_summary.py --output reports/phase4-lm1b-small-rust-dev-summary-v1.json
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo test --locked -p ncm-backend-cuda --features lm1b-experimental,cuda-cudnn --example lm1b_small_development_capture
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-backend-cuda --all-features --all-targets -- -D warnings
RUSTC_WRAPPER= ./tools/check.sh > reports/phase4-lm1b-rust-development-check-v2.log 2>&1
```

The first scorer launch lacked `transformers` and failed before scoring; the
successful runs used the existing pinned tokenizer environment through
`PYTHONPATH`. Two initial lint attempts found style and array-conversion errors;
the corrected capture passed all-feature CUDA lint. A focused test covers all
development seeds and rejects all official seeds; a direct seed-42 invocation
also failed before GPU initialization. The first standard check failed one
process-lifecycle test while other work was running; the isolated second check
passed, and the failure is retained in `reports/phase4-lm1b-rust-development-check-v1.log`.
Logs and data remain in ignored
`reports/`; no model weights or samples are tracked.
