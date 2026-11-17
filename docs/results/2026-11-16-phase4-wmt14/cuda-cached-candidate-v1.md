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
