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
cases. The source-bound full 3,003-row replay, frozen scorer and repeated alternating
performance/memory comparison remain pending. The capture and benchmark
supervisors require a clean tracked source tree and record source identities.
No WMT14 Metal or Phase 4 gate is claimed.
