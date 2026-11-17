# WMT14 uncached CFG eight-row diagnostic

An explicit experimental variant groups up to eight external rows per uncached
CFG call. The established four-row path retains its API and behavior; both
reject cached models and use no margin heuristic. The changed batch shape can
change floating-point results, so it requires independent qualification.

The real-CUDA regression matches separately invoked eight-row chunks for
external batches 1, 8, 9 and 16 and rejects invalid inputs and cached models.
Exact comparisons against the pinned Python full capture found no code or ID
differences in the first 35 rows, or in the separate 16-row ranges beginning
at 2176 and 2656 containing the previously difficult rows. Candidate capture
SHA-256 values in that order are
`6febb1ce55d57c3909d4e1c1d067c607c980b22e7f29aa422f219f63639308d5`,
`3940e71c84675adb404b0d41ff4305ff3836812be4f0653a0dec559525d84c3f`, and
`1a013dbe7464c0fc6c9c9f6d077a69452ead37b2336edb29057878619c3b9973`.
The comparator checks the explicit external-16/internal-8 identity and has
tests for changed outputs and wrong internal batching.

One nonexclusive batch-16 timing smoke with five warmups and 30 trials had
Rust median/p95 1.232480/1.236935 seconds, versus the earlier independent
Python median of 1.237539 seconds. This small difference is not a repeated
paired performance result or a gate pass. A clean-source full 3,003-row exact
replay, frozen scorer and repeated performance/memory comparison are pending.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= NCM_WMT14_CACHE_ARTIFACT="$PWD/artifacts/canonical/wmt14-inference-v1-final" NCM_WMT14_CACHE_PREFIXES="$PWD/reports/phase4-wmt14-tokenizer-reference-v1.json" LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo test --locked -p ncm-backend-cuda --release --features wmt14-experimental,cuda-cudnn --test wmt14_cached uncached_cfg_eight_matches_eight_row_chunks_and_rejects_cached -- --ignored
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" target/release/examples/wmt14_cfg_uncached_eight_capture artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json OUTPUT START COUNT
.reference/venv/bin/python tools/reference/compare_wmt14_cfg_eight_range.py reports/phase4-wmt14-cfg-cuda-full-v1/python.json CANDIDATE OUTPUT --candidate-sha256 CANDIDATE_SHA256
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" target/release/examples/wmt14_cfg_uncached_eight_perf artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json 16 1
.reference/venv/bin/python -m unittest discover -s tools/reference -p test_compare_wmt14_cfg_eight_capture.py
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-wmt14-uncached-eight-check-v1.log 2>&1
```

All-feature lint and the full CUDA check passed, including 18 device operator
cases. The original four-row real-CUDA chunk regression also passed after the
shared grouping refactor. The full-capture and benchmark supervisors require a
clean tracked source tree and record source identities. No WMT14 Metal or
Phase 4 gate is claimed.
