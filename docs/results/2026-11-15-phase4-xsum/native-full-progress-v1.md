# XSum reference-order CUDA full-run progress

The v2 quality capture resumes the two source-bound smoke shards with the
pinned 11,334-row dataset and Python reference. At this checkpoint, shard 0
completed all 5,667 rows: all 272,016 raw output IDs match, while one of
1,133,400 intermediate codes differs. The differing code is dataset row
2,690, level 6 position 20 (flat code position 100): Python selected 568 and
Rust selected 2,617. Its 48 raw output IDs match. The earlier v1 capture had
matched Python's code on this row. This difference must be investigated under
the frozen quality protocol; a targeted trace has been prepared locally.

Shard 1 is running; its first 320 rows matched all codes and output IDs. These
are partial results, not a full quality pass. The local outputs and manifests
are `reports/phase4-xsum-quality-native-verified-s0/` and
`reports/phase4-xsum-quality-native-verified-s1/`; they include the source,
artifact, executable, cuDNN and linked CUDA library fingerprints. The historical
v1 capture and the Python reference remain separate. Each finished batch is
published atomically and can be verified and resumed only under the same
fingerprints.

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_rust.py --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-native-verified-s0 --shard 0 --stop 11334 --profile v2
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_rust.py --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-native-verified-s1 --shard 1 --stop 11334 --profile v2
```

Do not treat the matching output IDs in this partial capture as a full XSum
quality pass. Full scoring, investigation of every intermediate-code
difference, repeated performance checks and the MLX gate remain pending.
