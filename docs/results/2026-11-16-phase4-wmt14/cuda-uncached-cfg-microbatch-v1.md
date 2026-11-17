# WMT14 uncached CFG microbatch candidate

This explicit experimental path runs the existing uncached CFG implementation
in sequential groups of at most four, returning host codes and IDs. It rejects
cached models, invalid prefixes, batch sizes outside 1–16 and invalid guidance
scales. There is no margin-triggered recomputation or cached CFG heuristic.
The ordinary cached CFG guard is unchanged.

The real-CUDA regression compares groups of 1, 4, 5 and 16 against independently
executed uncached groups of at most four and tests invalid inputs and the
cached-model rejection. A 35-row external-16/internal-4 capture matched the
retained Python batch-four codes and raw IDs exactly, including the partial
tail; capture SHA-256
`db58f201dec0bf37638cdbf88ccdc6ac6f2726ce08ab8f787c36f4e96f0dada0`.
These are bounded tests, not the full corpus or an unseen-input guarantee.

One batch-16 timing smoke, five warmups and 30 trials, produced Rust median/p95
1.26615/1.27121 seconds and host VmHWM 1,893,580 KiB. The earlier independent
Python batch-16 median was 1.23895 seconds; the smoke is slower and is not a
paired or full performance comparison. Desktop GPU activity may affect it.
Do not use this result as a speed or memory pass.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= NCM_WMT14_CACHE_ARTIFACT="$PWD/artifacts/canonical/wmt14-inference-v1-final" NCM_WMT14_CACHE_PREFIXES="$PWD/reports/phase4-wmt14-tokenizer-reference-v1.json" LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo test --locked -p ncm-backend-cuda --release --features wmt14-experimental,cuda-cudnn --test wmt14_cached uncached_cfg_microbatch_matches_exact_chunks_and_rejects_invalid_requests -- --ignored
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" target/release/examples/wmt14_cfg_uncached_microbatch_capture artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json reports/phase4-wmt14-uncached-cfg-microbatch-smoke-v1.json 0 35
.reference/venv/bin/python tools/reference/compare_wmt14_cfg_microbatch_capture.py reports/phase4-wmt14-cfg-supervisor-smoke-v1/python.json reports/phase4-wmt14-uncached-cfg-microbatch-smoke-v1.json reports/phase4-wmt14-uncached-cfg-microbatch-smoke-comparison-v1.json --reference-sha256 10d009a36f2a2282aa2d7bf1a530f60aa0b1b5f9eb58898069d0a3ddd3cea82f --candidate-sha256 db58f201dec0bf37638cdbf88ccdc6ac6f2726ce08ab8f787c36f4e96f0dada0
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" target/release/examples/wmt14_cfg_uncached_microbatch_perf artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json 16 1
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-wmt14-uncached-cfg-microbatch-check-v1.log 2>&1
```

All-feature lint and the full CUDA check passed, including 18 device operator
cases. The subsequent source-bound full replay at `6fc07a1` matched all 3,003
retained Python rows exactly: zero code or raw-ID differences. The candidate
capture SHA-256 is
`8a0ff957653084ad5d11da577a637e123307a47f4698921f415de647041a5483`,
and the comparison receipt SHA-256 is
`e1b4ae1dcabb918cfe31ff8809dc911c9b4fb1bd8d2ad660413f9f9dd17ab9e5`.
The frozen full and held-out scorer passed, reproducing the prior scored
result byte-for-byte in two independent scorer environments (SHA-256
`915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`).
This is a replay of previously inspected data, not an unseen-input proof.

The alternating three-process batch-16 comparison used five warmups and 30
trials per process. Every benchmark code and ID matched. Pooled median/p95
was **1.291226/1.296859 seconds Rust**, versus **1.237539/1.240858 seconds
Python**: this candidate is slower and does not pass the desired speed
direction. Python process medians were 1.231563/1.239367/1.237539 seconds;
Rust medians were 1.290281/1.293527/1.290831 seconds. Largest process VmHWM
was 2,143,576 KiB Python and 1,893,676 KiB Rust. Roughly one-second GPU
telemetry observed process maxima of 3,072 MiB Python and 1,640 MiB Rust;
these are sampled observations, not allocator peaks. All 279 telemetry samples
had successful queries, and other desktop GPU processes were present.
Benchmark manifest, measurements and telemetry SHA-256 values are
`40446553316f6263c12501e7dcf5b019d3b3b15fedf44d24d59ccb01f30935bb`,
`7391ded26ab160810201ac8a9fe3ba318f6458fb61e53ab03a352236310dfa61`, and
`6ffbd0e4c0fd1ffb837400520a9d8c96e024a91e3fa098408d300f677b49257a`.

The source-bound commands used clean tracked commit
`6fc07a143994da86095850d4c47731919880e30e`:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/capture_wmt14_cfg_uncached_microbatch.py --mode cfg1 --head 6fc07a143994da86095850d4c47731919880e30e --binary-sha256 e6fd19e5bc78fc183f0c4b8c3c11eb73a8e0864c62c1d368e0cd1e106701420f --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --rows 3003 --output reports/phase4-wmt14-uncached-cfg-microbatch-full-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_wmt14_cfg_uncached_microbatch_bench.py --output reports/phase4-wmt14-uncached-cfg-microbatch-bench-v1 --repetitions 3 --batches 16 --modes 1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/tokenizers-venv/bin/python tools/reference/wmt14_decode_capture.py reports/phase4-wmt14-uncached-cfg-microbatch-full-v1/rust.json reports/phase4-wmt14-uncached-cfg-microbatch-full-text-v1.json --capture-sha256 8a0ff957653084ad5d11da577a637e123307a47f4698921f415de647041a5483
.reference/wmt14-scorer-venv/bin/python tools/reference/wmt14_score_heldout.py reports/phase4-wmt14-cfg-cuda-full-python-text-v1.json reports/phase4-wmt14-uncached-cfg-microbatch-full-text-v1.json reports/phase4-wmt14-uncached-cfg-microbatch-full-v1/comparison.json reports/phase4-wmt14-uncached-cfg-microbatch-score-v1.json --reference-text-sha256 1dc421ee6bdd115e050c7d85069b7f7076c177881d35752181b0ee45ca7bf1df --candidate-text-sha256 df4c14dd6ae32f3b70b0a0a14c749d18cce0f7a23ec38bacd9dfb7954b5c4c29 --comparison-sha256 e1b4ae1dcabb918cfe31ff8809dc911c9b4fb1bd8d2ad660413f9f9dd17ab9e5 --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --mode cfg1
```

No WMT14 Metal or Phase 4 gate is claimed; the normal cached CFG guard remains.
