# XSum owned loader: Linux validation and performance

The uncommitted candidate consumes an already validated artifact and releases its
codec payload after codec upload, before generator upload. The borrowed constructor
remains available. Full host checks precede upload; this is a load-lifetime change,
not an artifact or numerical-tolerance change. The strict-CUDA CLI matched the
retained pinned Python reference on all 6,400 codes and 1,536 output IDs for 32
passages (`reports/phase4-xsum-owned-loader-cli-v1/result.json`). Strict CLI Clippy
and the new payload-release regression passed. Full `tools/check.sh --cuda` passed
with 18 real-CUDA operator cases using `setpriv --inh-caps=-all env RUSTC_WRAPPER=`;
log `reports/phase4-xsum-owned-loader-linux-v3.log`. The initial sandbox run failed because `sccache` was denied; a retry with
`RUSTC_WRAPPER=` could not access `nvidia-smi`. The first GPU-accessible run failed seven existing
identity tests because its process inherited a capability. Both failure logs are
retained. Clearing only that process's inheritable capability fixed the environment;
no machine-wide credential setting changed.

The CUDA performance example was then changed to use the candidate constructor.
The complete v2 collection used the pinned Python CUDA baseline, alternating three
fresh processes per runtime and batch, five warmups and 30 synchronized generations
per process, with matching checked codes and IDs in each batch. The first collection
was interrupted when an unrelated WMT14 compilation overlapped it; that partial v1
run remains in `reports/phase4-xsum-owned-loader-performance-v1/`.

| Batch | Runtime | Median of process medians ms | Pooled p95 ms | Median peak RSS MiB | Median sampled driver peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 158.89 | 160.41 | 1066.05 | 838 |
| 1 | Python | 192.78 | 194.48 | 1840.49 | 990 |
| 4 | Rust | 577.96 | 583.95 | 1064.27 | 1062 |
| 4 | Python | 704.31 | 712.17 | 1840.23 | 1624 |
| 32 | Rust | 4491.50 | 5760.65 | 1063.72 | 3080 |
| 32 | Python | 5489.13 | 5822.05 | 1847.82 | 6214 |

Timing excludes load; RSS includes it. One Rust batch-32 process had a 5220.78 ms
median versus 4491.50 and 4457.54 ms in the others, and pooled p95 is close to
Python's; the cause is not established. Driver memory is sampled and can miss
peaks; CUDA pool counters and PyTorch allocator counters cover different scopes.
There were 1,273 activity samples and no query failures; four other GPU processes
were observed, using 359–617 MiB in total. This is not proof of exclusive GPU use
or a paired measurement of the old loader. The results favor Rust in the stated
cells but do not prove that this candidate reduces Metal RSS.

Raw evidence is under `reports/phase4-xsum-owned-loader-performance-v2/`; manifest
SHA-256 `f3d20f3c53235d59272dcb05577f8c03eb848849fb704c1dc2403d7a24a539c2`.
Summaries are `reports/phase4-xsum-owned-loader-performance-v2-summary.json`
(SHA-256 `ddf98e88c88130efc4969b259ef4ea66ddbeddcad8cf074bb7a1884e25a0f9cc`)
and `reports/phase4-xsum-owned-loader-performance-v2-gpu.json` (SHA-256
`2b27717d95cc170d8d633733ef8b9559c61d0f3d105f5f5604cc3e24afcb4f2e`). The first
summary attempt failed because the driver reported fractional temperatures; the
summarizer now parses them as floats and a focused regression passes. No raw input
or numerical threshold changed. Full current-candidate quality capture, Metal
validation and the decision to retain the candidate remain pending.

The candidate's first full-quality shard subsequently completed with successful
source and linked-library postflight checks. Its 5,667 rows match the previously
scored Rust CUDA capture on every code and output ID. The second shard is still
running; this partial match is not a full quality result. The retained reference
was captured with driver 615.78.08; this capture uses driver 620.70.06, so the
actual output comparison, rather than driver identity, is the evidence here.

The second shard also completed with successful source and linked-library
postflight. A receipt-validated comparison of all 11,334 candidate rows with the
previously scored Rust CUDA capture found zero code and zero ID differences.
`reports/phase4-xsum-owned-loader-quality-v1-comparison.json` has SHA-256
`11bba60f4d27b9f1cdc1ff430286a513e4fc376e16efb30cee8b4c1a1335f8b0`; its two
candidate manifest hashes are `81d20ec08779503bb1bd251720e41fa9dc95ea1732d54ca31a53f81a62f23180`
and `602261e7399456bae7d15127f096308574c0e6f66075ec4045f057732dae851f`.
The retained capture's raw IDs matched Python and its full/held-out scores were
previously published; identical raw IDs imply identical deterministic evaluator
inputs. This is a source-bound output comparison, not a fresh scorer run. The
previous user-accepted stochastic CUDA exceptions remain unchanged, and Metal
validation of the owned loader remains pending.
