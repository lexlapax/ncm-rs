# XSum and WMT14 CUDA CLI text/decode cold-request screens

2026-11-20. The pre-change ID alias, current unified ID input, current `--text` and current `--text --decode` were measured on the same first retained corpus row per profile. This is an exploratory comparison of the changed CLI paths, **not** a release performance gate. Each condition ran 105 fresh processes, in three groups of five system warmups and 30 measured requests, with a rotating four-condition order. Wall time includes process startup, artifact/tokenizer load, generation, readback and JSON writing; GNU time supplies peak process RSS. Warmups do not keep the model resident. All 420 stdout streams per profile matched the respective previously checked ID or decoded output hashes. The corpus-wide text/ID and retained-output decoder checks are [separate](cuda-text-input-v1.md#full-corpus-cuda-textid-generation-retest).

| Profile | Condition | Median ms | Pooled p95 ms | MAD ms | Median peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| XSum | Old ID alias | 1272.521 | 1283.847 | 4.957 | 989.2 |
| XSum | Unified ID | 1274.031 | 1288.735 | 6.021 | 989.3 |
| XSum | Unified text | 1289.175 | 1301.184 | 5.505 | 997.4 |
| XSum | Unified text + decode | 1290.179 | 1305.998 | 5.247 | 997.4 |
| WMT14 | Old ID alias | 1842.991 | 1885.483 | 9.954 | 1463.9 |
| WMT14 | Unified ID | 1840.519 | 1881.375 | 7.257 | 1464.0 |
| WMT14 | Unified text | 1915.573 | 1949.289 | 7.728 | 1472.2 |
| WMT14 | Unified text + decode | 1913.785 | 1958.635 | 8.603 | 1472.2 |

Descriptive paired median differences in milliseconds with 95% bootstrap intervals (10,000 resamples of 90 within-ordinal pairs, Python RNG seed 42):

| Profile | Unified ID − old ID | Text − unified ID | Text+decode − text |
| --- | ---: | ---: | ---: |
| XSum | +2.248 [−0.420, +5.650] | +14.766 [+12.404, +17.890] | +1.755 [−1.855, +4.091] |
| WMT14 | −2.304 [−9.728, +2.436] | +74.469 [+70.967, +78.871] | −0.701 [−3.996, +2.150] |

These observed text-path increases include tokenizer/asset preparation as well as encoding; the run does not isolate their cause. The intervals do not prove independent samples or remove system drift. This is one row and one batch size per profile; no general regression threshold is established. The desktop GPU was shared; 539 XSum and 790 WMT14 system-wide roughly one-second samples are not per-process native allocation peaks and may miss short peaks. There is no same-request Python comparison, warm in-process latency, full batch sweep, native CLI device peak or Metal evidence.

The baseline binary SHA-256 was `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`, candidate `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`; artifact/reference and expected stdout hashes are pinned in the runner. Runner SHA-256 `b1c86c4bfb3b5828e55173e4a2c21a606141670dcd906bd6ac92d6500f6c1fc2`. It watches those files and selected libraries before/after, not every mapped library. The [sanitized trial series](cuda-text-cold-trials-v1.json), SHA-256 `d2796b402425896ee3bef2e0fef4a93a10414170ea363c814b8f87d8a9b5ddfd`, preserves measured timing/RSS arrays and paired summaries, not inputs, outputs, paths, warmups or GPU telemetry. The exporter independently rechecked every retained stdout hash, RSS file, trial order, summary and GPU CSV hash; two exports matched byte-for-byte. Offline controls and `setpriv --inh-caps=-all tools/check.sh` passed. Raw bundles remain ignored under `reports/phase5-{xsum,wmt14}-text-cold-v1/`, not durably published. No inference code changed.

Reproduce with the pinned artifacts, binaries and corpus references from the [text screen](cuda-text-input-v1.md), using fresh paths:

```sh
for profile in xsum wmt14; do
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
    python3 tools/bench/phase5_text_cold_screen.py --profile "$profile" \
    --baseline reports/phase5-cli-equivalence/baseline-ncm --candidate target/release/ncm \
    --artifact-root artifacts/canonical --reference "reports/phase5-text-corpus-v1/$profile.json" \
    --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
    --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
    --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
    --output "reports/NEW-$profile-text-cold"
done
# Audit the original retained bundles and reproduce the published sanitized data:
setpriv --inh-caps=-all python3 tools/bench/export_phase5_text_cold.py --reports reports --output reports/NEW-text-cold-trials.json
```

The export command intentionally pins the original raw bundle hashes; it is not a generic verifier of new runs. The previously recorded XSum tie and WMT14 CFG=1 diagnostic failure remain unchanged.
