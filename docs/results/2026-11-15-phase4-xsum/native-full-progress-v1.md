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

Both shards subsequently completed and passed their source/library postflight.
Across all 11,334 rows, all 544,032 raw IDs match Python; exactly one of
2,266,800 intermediate codes differs. Shard 1 has no code or ID differences,
including the formerly failing batch. The two recorded manifest hashes are
`8250d6b1b41d991057327ab8b205a84a7f2eabd70485c8770d21f0b474c49539` and
`375cec5e2d1b4c840be1fb2ee6ef9b78af188971653d242238c9a8f0a9bbf5dd`.

A real-CUDA replay restored the saved Python random state and reproduced every
saved Python code. At row 2,690, level 6 position 20, Python's scores for
codes 568 and 2,617 are exactly tied (both FP32 bits `1057373626`); its sort
selected 568. The native Rust replay reproduced every saved Rust code and raw
ID, with corresponding scores `1057373627` and `1057373628`, selecting
2,617. Both selected scores are finite. The difference is therefore a
one-FP32-step Rust ordering at an exact Python maximum tie; all final IDs on
that row match. This evidence explains the recorded choice but does not define
universal tie ordering. Logs and trace inputs are retained in
`reports/phase4-xsum-tie-probe-v1/`.

The full output capture is complete, but scored ROUGE and GPT-2 perplexity,
the frozen-rule comparison, repeated performance and MLX remain pending.
