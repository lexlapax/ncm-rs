# XSum CUDA guidance on real inputs

At CFG scale 1, the first eight padded XSum prefixes matched pinned Python CUDA
on all 1,600 intermediate codes and 384 penalized output IDs. The input is the
first eight rows of the previously verified 32-row development fixture; its
SHA-256 is `beeb2c1fb6d46b0cdf3cfad61c3f9a6cd32a26db5099b6fe6342862edffdef36`.
This supplements the two synthetic scale-1 and scale-2 comparisons; it is not a
full quality evaluation at positive guidance. The Python trace and Rust JSON
are in `reports/phase4-xsum-cfg-real-reference-v1/`, with hashes
`fe20849f9da1d5c679a4cb5ad36e69f504c9eaa26a2981ec9b93177acec4186b` and
`810975262ea338db24c83048c7d4ae538bac7a225032b58d36110efc1d45ffba`.
Negative and NaN scales failed explicitly in separate real-CUDA CLI checks.

An exploratory scale-1 comparison used the same pinned dataset fixture, two
warmups and five timed generations per process. All codes and IDs matched for
each batch and runtime. The median timings exclude load; `/usr/bin/time` peak
RSS includes the whole process. Python garbage collection was run after each
trial and excluded from the timings, as in the full CFG-zero benchmark.

| Batch | Rust median ms | Python median ms | Rust peak RSS MiB | Python peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 286.83 | 357.44 | 1064.30 | 1864.28 |
| 4 | 1068.44 | 1309.11 | 1064.21 | 1857.70 |

These single-process screens show the requested speed and process-memory
direction for the tested inputs, but they are not repeated performance gates.
No activity telemetry or linked-library fingerprint was collected for this
screen, so desktop interference cannot be assessed. Raw results, process
memory reports and source/binary/input fingerprints are under
`reports/phase4-xsum-cfg-performance-v1/`; the listed fingerprints matched
after collection.

The reference uses `tools/reference/xsum_cfg_trace.py --cfg-scale 1.0
--prefix-json INPUT --decisions-only` with the pinned `--upstream`, `--source`
and a new `--output`; `xsum_cfg_compare.py` compares its trace with CLI output.
The two timing programs accept CFG as the final argument `device 5 1.0` for
`xsum_perf_screen`, and `--no-stage-sync --collect-garbage --trials 5
--cfg-scale 1.0` for `tools/reference/xsum_perf_screen.py`. Both use
`reports/phase4-xsum-real-inputs-v1/prefixes.json`, the v2 artifact for Rust,
and the pinned upstream checkpoint for Python. Run separately with
`OMP_NUM_THREADS=1` and `/usr/bin/time -v`; the native binary needs the local
cuDNN library path. The scripts and binary are fingerprinted in the raw report.
