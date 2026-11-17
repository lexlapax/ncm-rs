# WMT14 CUDA prefix-cache candidate

On source `2e9bf9b`, the explicit cache candidate ran the frozen 3,003-row
base and CFG comparisons against retained Python outputs. This is a retest on
previously inspected data. Base matched all 939,939 codes and 192,192 raw IDs;
the pinned scorer passed both full and held-out rules. An independent offline
scorer replay matched byte-for-byte (SHA-256
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`).
Full-set BLEU was 21.7202, lowercase BLEU 22.6121, chrF 49.5649 and length
ratio 0.9374 for both; identical hypotheses yield zero paired intervals under
the frozen shortcut.

The CFG cache **failed** the unchanged exact-ID gate: nine codes and eight
raw IDs differ across rows 2181 and 2658. All eight IDs occur after the common
first EOS; decoded text and quality metrics match, but that does not waive the
gate. The scorer and independent replay both record failure, byte-identically
(SHA-256 `e00012f8470326ec29920e07ae7271f9035bc85e6e54d31ddcdbbc1b54242692`).
The prior no-cache CFG reference-order path passed its own full retest and is
not changed by this experiment. The base comparison and CFG comparison hashes
are `70c5dca7ed8d9f097eb5a1a9f6b2df980430eaf21fa1456e93253dd9ec0d27c4`
and `71ce8bfd5517eec82d456ace965f73c24a5e5599e60162076a4b02637a03a86b`.
Raw captures and scorer outputs remain in ignored `reports/`.

A separate base benchmark used the same pinned Python baseline and prefixes,
three alternating fresh processes per runtime, five warmups and 30 measured
batches per process. All benchmark codes and IDs matched. The table reports
pooled median/p95 ms per batch, peak host RSS, and the largest sampled process
GPU memory (approximately one-second sampling; not allocator peaks).

| Batch | Python median / p95 ms | Cached Rust median / p95 ms | Speedup | Host MiB Python / Rust | Sampled GPU MiB Python / Rust |
|---:|---:|---:|---:|---:|---:|
| 1 | 64.14 / 64.53 | 38.11 / 38.29 | 1.68× | 2094 / 1850 | 1360 / 1222 |
| 4 | 178.34 / 179.83 | 84.35 / 84.70 | 2.11× | 2093 / 1844 | 1492 / 1478 |
| 16 | 650.18 / 653.20 | 283.33 / 284.64 | 2.29× | 2093 / 1844 | 2300 / 2854 |
| 32 | 1276.09 / 1277.92 | 547.60 / 548.69 | 2.33× | 2093 / 1880 | 3726 / 4488 |

At the pooled median cached Rust achieved 26.24/47.42/56.47/58.44 samples/s,
1679/3035/3614/3740 output positions/s, and 210/889/1359/1424 non-special
pre-EOS token IDs/s at batches 1/4/16/32 respectively. These counts use fixed
prefixes and 64 output positions per sample. Python cleanup-inclusive medians
were 114.59/229.61/701.90/1327.74 ms. CUDA-event medians for Rust were
38.05/80.28/259.83/499.91 ms; event spans can include host-induced stream idle
and exclude final readback. Python allocated/reserved peaks were
1126/1192, 1245/1324, 1724/2132 and 2360/3558 MiB. Rust's sampled GPU
memory exceeds Python's at batches 16 and 32, so the desired memory direction
is not met. Device memory scopes are not identical.

The telemetry contains 411 samples and no query errors; desktop GPU processes
were present, and the run was not exclusive. Stage breakdowns for the cached
path, independent physical capacity and exclusive-device results remain
pending. The manifest, measurements and telemetry SHA-256 values are
`d9c642e0bbcf902df749f33583b7560bec71c024d9b0f9de58bdee4c9a91bea6`,
`ac85d816f39bbc792e26dcecb8bc6ce93919ecca974e40d44f4595eb87755da4`, and
`91d203f17c5df648aaead51e014e3083e7cc0ee96984100f92a8e0db82c38407`.
The known CFG failure must remain explicit; a base-only guard and further
investigation are pending. No Metal WMT gate is claimed.

A follow-up explicitly rejects guided generation from the ordinary cached
constructor, directing callers to the no-cache model. A separately named
`cached_cfg_diagnostic` constructor remains available solely to reproduce and
investigate the documented failure. A real-CUDA regression confirms the guard,
small diagnostic cases, and existing forced-history bands (log SHA-256
`ecee314c26d80ef5b9a4b9c711e5890ae07eb8be3ebd6e242e3851d6ce536050`).
The first full guard check failed because an untracked microbatch benchmark
example was auto-discovered without its feature declaration; the failed log
is retained at `reports/phase4-wmt14-cache-guard-cuda-check-v1.log`. The
feature declaration was added and a full rerun is pending.

A separate explicit microbatch experiment will process external base batches
in sequential groups of at most four, measuring whether it reduces peak GPU
memory while maintaining exact outputs and a speed advantage. It is not yet a
production scheduling policy; full source-bound output and repeated
performance/memory evidence remain pending.

The corrected full CUDA check passed, including 18 operator cases; its log
SHA-256 is `b05b70c6a3194c9d789d56ace4a5fcc025d69e51457c689788ef42650e1f0d62`.
A batch-32 microbatch smoke matched all retained Python codes and IDs and had
a single-process median of 655.55 ms (result SHA-256
`6c6cc0aaf7afb245cba75b655f6ef4b16507a5bb8a689c6dd278f3f0c977c2d3`).
This smoke does not establish peak GPU memory or a repeated performance gate.

## Explicit batch-of-four scheduling experiment

At `915515a`, the separate benchmark processed external base batches in
sequential cached groups of at most four. All codes and raw IDs matched the
same-batch Python output across three alternating processes, five warmups and
30 measured batches per process. The same 3,003-row base cache retest above
used groups of four and matched every reference output. This benchmark is an
explicit scheduling experiment, not yet a production API or a fresh corpus.

| External batch | Python median / p95 ms | Microbatched Rust median / p95 ms | Host MiB Python / Rust | Sampled GPU MiB Python / Rust |
|---:|---:|---:|---:|---:|
| 16 | 647.84 / 652.16 | 335.57 / 336.91 | 2093 / 1844 | 2300 / 1606 |
| 32 | 1278.08 / 1351.89 | 683.52 / 711.26 | 2093 / 1844 | 3726 / 1734 |

Rust was 1.93× and 1.87× faster at the pooled median, with lower sampled host
and GPU process memory for these workloads. Individual process medians were
329.97/335.65/336.39 ms at batch 16 and 667.94/686.50/693.16 ms at batch 32;
Python's were 643.98/647.84/651.70 and 1277.04/1277.35/1303.82 ms. This
variation and the high batch-32 p95 are reported rather than hidden. The
CUDA-event spans include intermediate host readbacks and idle time, so they
are not pure GPU execution. There were 346 telemetry samples with no query
errors; desktop GPU processes remained present. The measurement is not an
exclusive-device gate and sampled driver memory is not an allocator peak.
Manifest, measurements and telemetry SHA-256 values are
`569a90153da587c77bb282ccc7306ba7b8d71bd3e5f6a7251ef41a3f0cab7928`,
`1d1052ab3694f4bc76ede56b3214cbeb937c8a49a608d97d477f58a86611b6ed`, and
`ec072880e88c67e41cff25bb2357fcb4e3be17fccf3dc256202b40b7e24addfe`.
Raw files are retained under ignored `reports/phase4-wmt14-cache-microbatch-perf-v1/`.

An explicit experimental base-only host-output API now runs the same
batch-of-four schedule and rejects unsupported backend/input combinations.
A real-CUDA test compared its codes and IDs to the no-cache model at external
batches 1, 16, 17 and 32, including a partial tail; it also checks invalid
requests and the CFG guard. All three targeted cache tests passed (log SHA-256
`c94b2811e1743bb450049962c445b33fcfde34b8f595b06f1f346b7cf02d0cc9`).
The first strict Clippy run caught an unused import in the updated example;
a corrected rerun passed. Full CUDA regression and source-bound API replay and
performance measurement remain pending. The previous benchmark used the same
scheduling as an example loop and is not presented as a measurement of the
new API binary.
The API's full CUDA check passed 18 operator cases (log SHA-256
`f393bb541cef0ed4bb572b5c3f7c4811f3ba201e299e20f9b44a393109ffb1ee`).
The new source-bound capture example and updated API benchmark example passed
strict feature-specific Clippy and release build. A local 35-row capture using
external groups of 32 and internal groups of four matched all Python codes and
IDs, including the three-row external tail; SHA-256
`f21f3314085d1e729c3c113e23376876a492e7f580744979cd15711a49c7158c`.
Full source-bound capture and repeated API benchmark remain pending.
