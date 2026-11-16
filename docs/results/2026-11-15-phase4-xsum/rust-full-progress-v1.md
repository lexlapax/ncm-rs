# XSum Rust CUDA quality progress

The source-bound Rust quality run is resuming the previously validated
development batches into the full XSum dataset, with the two original
stride-two shards run sequentially. At the recorded checkpoint, the first 133
batches of shard 0 covered 4,256 rows and matched the pinned Python reference
on all 851,200 codes and 204,288 raw IDs. This is a partial progress check,
not a full quality comparison or a Linux gate.

The first shard subsequently completed all 5,667 rows. The read-only progress
audit checked all 178 batches against Python and found zero differences among
1,133,400 codes and 272,016 raw IDs. Its ordered batch inventory SHA-256,
computed using the method in the Python reference record, is
`faea501c63c4afe9b60eb5e0dde5cf5e5a3dde57d03da0078d63a8d053b5481e`.
The second shard is still running; this is not a full quality pass.

Rust shard outputs are retained under
`reports/phase4-xsum-quality-rust-dev-v2-s0/` and
`reports/phase4-xsum-quality-rust-dev-v2-s1/`; the full Python reference and
its integrity hashes are in [the reference record](python-full-reference-v1.md).
The Rust manifest SHA-256 values are
`6884e2a1bc1a6a05b1c9ac825834f758583124f973d215b248133a207903d1b3` and
`609615cb9a307232834463c473cab7a9e6549ff3a0d3705810dd83299ec30af8`.
The strict CUDA batch executable SHA-256 is
`1f84f6c768967b307a2c268532834b2435427d33386d31895b011185134911d2`.
The manifests record the dataset, binary, relevant sources and native library
hashes; a subsequent source and binary check matched them.

The run was launched with the commands below. Each completed batch is
published separately, and restarting the same command verifies existing
receipts and continues from the next batch. Use the same source, binary,
libraries and GPU/driver identity; a mismatch fails explicitly.

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_rust.py --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-rust-dev-v2-s0 --shard 0 --stop 11334 > reports/phase4-xsum-quality-rust-full-s0.log 2>&1
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_rust.py --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-rust-dev-v2-s1 --shard 1 --stop 11334 > reports/phase4-xsum-quality-rust-full-s1.log 2>&1
.reference/venv/bin/python tools/bench/audit_phase4_xsum_progress.py --shard 0
```

The last command is a read-only progress aid, not the final
quality comparator. Full Rust generation, held-out and full scoring, and the
frozen-rule comparison remain pending.
