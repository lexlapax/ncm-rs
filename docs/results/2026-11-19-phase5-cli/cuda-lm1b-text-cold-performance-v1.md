# LM1B CUDA CLI default-sidecar and empty-text cold screens

2026-11-20. For small, medium and large, the old ID alias with an explicit rotary table was compared with unified ID/default-sidecar, empty `--text`/default-sidecar and empty `--text --decode`/default-sidecar requests. All used seed 42 and one row; empty text maps to `[101]`. Each condition ran 105 fresh processes in three groups of five system warmups and 30 measured requests, rotating condition order. Every one of the 420 stdout streams per size matched its pinned undecoded or decoded output hash. The [one-seed cohort routing](cuda-lm1b-cli-cohort-v1.md), [empty-text policy](cuda-lm1b-empty-text-v1.md) and [decoder](cuda-decode-screen-v1.md#lm1b-retained-cuda-cohort-decoding) evidence are separate.

Wall time includes process startup, artifact/rotary/tokenizer loading, generation, readback and JSON writing; warmups do not keep the model resident. Median and nearest-rank p95 pool 90 requests; RSS is median process peak. This is an exploratory shared-desktop screen, **not** a release gate.

| Size | Condition | Median ms | p95 ms | MAD ms | Median peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Small | Old ID, explicit rope | 1569.553 | 1585.645 | 6.543 | 1209.7 |
| Small | New ID, default rope | 1570.792 | 1597.435 | 6.752 | 1206.1 |
| Small | Empty text | 1569.932 | 1594.820 | 6.563 | 1206.0 |
| Small | Empty text + decode | 1578.256 | 1600.746 | 5.135 | 1213.1 |
| Medium | Old ID, explicit rope | 2738.966 | 2758.405 | 8.146 | 2048.9 |
| Medium | New ID, default rope | 2734.074 | 2763.009 | 9.973 | 2044.2 |
| Medium | Empty text | 2738.192 | 2756.894 | 8.899 | 2044.1 |
| Medium | Empty text + decode | 2743.683 | 2764.608 | 8.602 | 2051.1 |
| Large | Old ID, explicit rope | 4878.087 | 5025.548 | 30.238 | 3605.5 |
| Large | New ID, default rope | 4890.339 | 5035.057 | 39.338 | 3603.4 |
| Large | Empty text | 4883.000 | 5002.159 | 31.769 | 3602.8 |
| Large | Empty text + decode | 4894.270 | 5099.522 | 32.204 | 3609.4 |

Descriptive paired median differences, milliseconds [95% bootstrap interval], resampling the 90 within-ordinal pairs 10,000 times with `random.Random(42)`:

| Size | New ID − old ID | Empty text − new ID | Text+decode − text |
| --- | ---: | ---: | ---: |
| Small | +2.964 [−1.503, +6.379] | −1.378 [−4.803, +2.606] | +7.745 [+4.337, +11.197] |
| Medium | −3.563 [−9.605, +1.589] | −0.696 [−3.909, +5.877] | +7.858 [+3.252, +12.003] |
| Large | +3.153 [−0.097, +6.488] | −3.532 [−10.033, +5.638] | +8.885 [+4.666, +17.468] |

Decode includes BERT tokenizer setup and decoding; this experiment does not isolate their contributions. Large timings were visibly more dispersed. Intervals do not prove independent observations, remove drift, or establish a regression budget. The roughly one-second system-wide GPU monitor recorded 661/1,150/2,059 samples, not per-process native peaks; no exclusive GPU reservation, Python comparison, warm-model timing, batch sweep or Metal measurement was established.

The old/current binary SHA-256 values were `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098` / `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`; manifests, sidecar and expected output hashes are pinned in the runner. The runner SHA-256 is `29ab48ea94e4e16c72236575aa5aac5bfc9bb3e84f5e040c74c667003fbc7846`. It hashes those identities and selected CUDA libraries before/after, not all mapped libraries. The [sanitized trial series](lm1b-text-cold-trials-v1.json), SHA-256 `216282dce10ee35e5c6726c0789e3d6388b7f7b0b5560942014b11079bb4d11e`, includes measured timing/RSS arrays and paired summaries, not inputs, outputs, paths, warmups or telemetry. Its exporter rechecked all retained stdout hashes, RSS files, order, summary and GPU CSV hashes and reproduced identical bytes twice. Runner/exporter controls, three-size CUDA smoke and `setpriv --inh-caps=-all tools/check.sh` passed. Full raw bundles remain ignored under `reports/phase5-lm1b-{small,medium,large}-text-cold-v1/`, not durably published. No inference code changed.

Reproduce with pinned binaries/artifacts and sidecar, using fresh output paths:

```sh
for size in small medium large; do
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
    python3 tools/bench/phase5_lm1b_text_cold.py --size "$size" \
    --baseline reports/phase5-cli-equivalence/baseline-ncm --candidate target/release/ncm \
    --artifact-root artifacts/canonical \
    --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
    --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
    --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
    --output "reports/NEW-lm1b-$size-text-cold"
done
# Audit the original retained bundles and reproduce the published sanitized data:
setpriv --inh-caps=-all python3 tools/bench/export_phase5_lm1b_text_cold.py --reports reports --output reports/NEW-lm1b-text-cold-trials.json
```

The exporter intentionally pins these original raw bundle hashes; it is not a generic verifier for newly collected runs.
