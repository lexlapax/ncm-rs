# WMT14 CUDA deferred selection-validation candidate

The [batch-32 comparison](cuda-performance-v1.md) found the experimental
reference backend slower than Python. The generator currently reads selected
codes to the host at every level to reject tied or nonfinite maxima. This
candidate uses the already supported sentinel-safe embedding on generated
codes, retains the original indices unchanged, then reads and rejects any
sentinel in the concatenated codes before decoding or returning. Forced-code
embedding is unchanged. Invalid rows are temporarily mapped to a valid
placeholder solely for safe intermediate device execution; the sentinel is
never accepted as output. Backends without the safe embedding fail explicitly.

A real-CUDA regression constructs unique, tied and NaN rows, checks the safe
embedding and confirms that the original invalid sentinels survive
concatenation. It passed with the experimental/cuDNN feature set. The full
CUDA check, full base and CFG exact-output replay, frozen quality scoring and
repeated performance/memory comparison remain pending. The first bounded
batch-32 timing/output smoke is diagnostic only. Prior failures and retest
limitations remain documented; no oracle or tolerance changes are authorized.

The batch-32 CFG=0 smoke reproduced the previous Rust codes and raw IDs exactly;
its single-process median was 1.276 s, which does not establish a speedup.
The local full CUDA check passed 18 device cases (log SHA-256
`483287251448c6e952ea510bb5393a86cad1159d1924ae6abafe913b6c69bdc5`).
The targeted sentinel test log SHA-256 is
`25cc6516ae17ce7f59c727cbabd25123fc5f15cb565388b847da5296ff55b1d4`.
The additional experimental-feature regression and full candidate replays
remain pending.
The release regression with `wmt14-experimental,cuda-cudnn` also passed; its
log SHA-256 is
`f48a337da159a50186f6727a5caa77a8dd6231970517be1fbd5e833a1b43d2a5`.

## Full replay and performance result

On committed `0027d8b`, both source-bound 35-row smokes passed, followed by
both 3,003-row comparisons against retained Python outputs. All 939,939 codes
and 192,192 raw IDs matched per mode. The base Rust capture is byte-identical
to the previously scored Rust base capture (SHA-256
`d26b8a3b16a4ad07565f52d3e61f22b30e2512ea10348e674a24e9ed52fecc79`),
and the CFG capture is byte-identical to the previously scored CFG candidate
(SHA-256 `ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f`).
The prior independently replayed scores and zero paired differences therefore
apply. This is a retest on previously inspected data, not a fresh held-out
study. The full comparison hashes match those in the prior candidate report.

The repeated synchronized comparison used three alternating fresh processes,
five warmups and 30 measured trials each, with output equality at every
workload. Times below are pooled median/p95 milliseconds per batch; the prior
Rust median is from the separate earlier run, so run-to-run drift limits the
interpretation of small changes.

| CFG | Batch | Python median / p95 | Candidate Rust median / p95 | Prior Rust median |
|---:|---:|---:|---:|---:|
| 0 | 1 | 64.74 / 65.18 | 60.05 / 60.39 | 59.78 |
| 0 | 4 | 178.98 / 180.37 | 177.16 / 177.71 | 175.60 |
| 0 | 16 | 652.23 / 654.36 | 661.33 / 662.88 | 661.71 |
| 0 | 32 | 1276.72 / 1278.81 | 1300.78 / 1303.07 | 1299.63 |
| 1 | 1 | 102.17 / 102.98 | 98.77 / 99.33 | 98.80 |
| 1 | 4 | 330.90 / 331.77 | 321.80 / 322.52 | 322.45 |
| 1 | 16 | 1237.78 / 1239.20 | 1222.05 / 1224.05 | 1223.69 |

Peak host RSS was 1,839–1,875 MiB for Rust and 2,093–2,094 MiB for Python.
Approximate one-second sampled process GPU peaks (Python/Rust MiB) were
1360/1222, 1492/1574, 2300/2630, 3726/4072 for CFG=0 batches 1/4/16/32;
and 1364/1222, 1538/1576, 3072/2600 for CFG=1 batches 1/4/16. These are
sampled driver values, not allocator peaks. Four desktop GPU processes were
present, and the full and batch-32 telemetry contained 656 and 287 samples
respectively, with no query errors. The comparison remains nonexclusive.

The candidate did not show a meaningful improvement over the prior Rust run,
and Rust remains slower than Python at CFG=0 batches 16 and 32. The immediate
per-level validation was restored byte-for-byte to the pre-experiment generator;
the sentinel regression is retained. The full restored-path CUDA check passed
18 operator cases (log SHA-256
`dd908881da9c9312ff8e78c0720be7ee67972d1b5dc8981ab2389ad9c2c801a4`).
The WMT CUDA performance gate remains open. Raw evidence is under ignored
`reports/phase4-wmt14-deferred-{base-full,cfg-full,perf-full,perf-b32}-v1/`.
The full/batch-32 performance manifests hash to
`74eaebf12b518cfed6693ef2998316eceb1d8134782ea94f8e62d77a3af58149` /
`9b84766c0e27e41e56393cecdea6633f842adaa8e7f1732cfe69f9a035e6fcf8`;
measurements hash to
`d49a2a4a83b55d6620d56d35f00b097c9927148a0c85bc6d07faabdd4f21d203` /
`77664027dbd45bbf0fc9359eecd5b0350e364e3e0d9b3a183f2b641e19b1e459`.
