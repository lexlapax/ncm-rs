# Sudoku medium/hard re-export CUDA cold-request screen

2026-11-20. The original and user-approved provenance-only candidate artifacts were compared with the **same** CUDA CLI binary and `sample` command on one two-board request per difficulty. This is a bounded artifact comparison, not a release performance gate. The [full-corpus output comparison](cuda-sudoku-reexport-corpus-v1.md) and [byte-level artifact comparison](linux-export-repro-v1.md#byte-level-comparison-of-alternate-exports) remain separate evidence.

Each condition ran 105 fresh processes: three alternating groups with five system warmups and 30 measured requests per group. Wall time includes process startup, artifact load, generation, output readback and JSON writing; `/usr/bin/time` reports process peak RSS. All 210 outputs per difficulty matched the expected stdout hash. Warmups are not in-process warm model runs. The median paired differences and descriptive 95% intervals below bootstrap the 90 within-group pairs 10,000 times using Python `random.Random(42)`; this does not establish independence or a regression threshold.

| Difficulty | Artifact | Median ms | Pooled p95 ms | MAD ms | Median peak RSS MiB | Paired candidate − original median ms [interval] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Medium | Original | 529.438 | 544.886 | 5.672 | 587.965 | +0.100 [−3.709, +3.917] |
| Medium | Re-export | 528.338 | 548.238 | 5.784 | 587.904 | same pairs |
| Hard | Original | 530.964 | 546.141 | 5.217 | 588.006 | +1.966 [−1.937, +6.916] |
| Hard | Re-export | 532.253 | 549.290 | 6.502 | 587.936 | same pairs |

These observations do not prove a non-regression margin. The desktop GPU was shared and not reserved; `nvidia-smi` logged 112 system-wide samples per run, not per-process native peaks. There is no Python comparison, stage split, larger batch, isolated-GPU claim or Metal evidence. The CUDA screen alone does not qualify the artifact or complete Phase 5.

The binary SHA-256 was `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`. Original and candidate manifest hashes are retained in the linked full-corpus report. The first two rows of each validated 2,000-row input were serialized as compact JSON with a newline. Medium input/output SHA-256: `8ce8ac13af0fddc753e55ee06ab2898647c5a652b931754e701a715ad60a57b7` / `b5a22993f2df60ece95b6b188e009a55cf10ad59a2591e69e478a025125d104d`; hard: `355ed0c09c83ca5cb3c67fcb5bd77057e2ccdcc7a05f054322ebcba283925ece` / `655dbfdb50d1210923246a730a08332156df64418365328d55a2f20af8b7caf0`.

The extended runner SHA-256 was `b5f71152912501963d701517ba2902f9c93f3d74e506524ba5a2a34febf3afdc`. It pins both manifests and the same binary, alternates the artifacts, hashes watched inputs before/after, and rejects distinct binaries or identical artifact identities in this mode. Unit tests and `setpriv --inh-caps=-all tools/check.sh` passed. Ignored raw result hashes are medium `5ab7bedfde94325862330da69fd7b399e809115131042ec498d33fa204950d60` and hard `c43de257621f3be5d12acc9f3f1780493610a4fa80c2df2b4e881b7ec44b225b`; their GPU CSV hashes are `c85c8157bacfb8e0dc658e9dd4780554b09240f9c9ed7034c7e7d28ed26b821b` and `ab183c4681a8b2b6b836784083ddf80a623c560a4520f7cef659e49273519e06`. The evaluation-derived inputs and raw output bundles remain local under `reports/`.

To reproduce, prepare the first-two-row inputs and their hashes as above, build the identified CLI and pass each pair of original/candidate manifests and expected output hashes from this report and the full-corpus report:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
  LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
  python3 tools/bench/phase5_cli_cold_screen.py \
  --baseline target/release/ncm --candidate target/release/ncm \
  --baseline-sha256 BINARY_SHA --candidate-sha256 BINARY_SHA \
  --baseline-artifact ORIGINAL_DIR --baseline-manifest-sha256 ORIGINAL_MANIFEST_SHA \
  --artifact REEXPORT_DIR --manifest-sha256 REEXPORT_MANIFEST_SHA \
  --input TWO_ROW_INPUT --input-sha256 INPUT_SHA --expected-output-sha256 OUTPUT_SHA \
  --watch-file /usr/lib/libcuda.so.1 --watch-file /opt/cuda/lib64/libcurand.so.10 \
  --watch-file /opt/cuda/lib64/libcublas.so.13 --watch-file /opt/cuda/lib64/libcublasLt.so.13 \
  --watch-file "$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib/libcudnn.so.9" \
  --output reports/NEW-DIRECTORY
```

The pairing and bootstrap procedure is described in the [Sudoku easy cold screen](cuda-sudoku-cold-screen-v1.md). The watcher hashes only the specified libraries, not all loaded libraries; no raw bundle is durably published.
