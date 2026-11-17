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

## Explicit microbatch API replay

At `4fbe4e6`, the source-bound API capture used external groups up to 32 and
internal groups of four. Both its 35-row smoke and complete 3,003-row replay
matched every reference code and raw ID. The full capture SHA-256 is
`9191e04fbf50328d207a378f8171379cac8a543767adea5eb8db55059af6fa98`;
the comparison SHA-256 is
`f40a1e374f6af91e583ca9831d36943cf746b8a9f3f7067f54f0a4bec0459b88`.
The frozen base scorer and independent offline replay passed byte-identically
with SHA-256 `84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`.
This reuses previously inspected rows and is not a new blind study.

The committed API benchmark repeated the external batch-16 and batch-32
comparisons with three alternating processes, five warmups and 30 trials per
process. Every code and raw ID matched. Pooled median/p95 milliseconds were
651.06/653.39 Python versus 336.55/337.73 Rust at batch 16, and
1279.47/1281.18 Python versus 670.59/672.27 Rust at batch 32. Process medians
were 333.60/336.87/336.70 ms Rust and 649.57/650.99/652.59 ms Python at batch
16; 668.14/671.43/670.79 and 1277.98/1279.47/1280.28 ms at batch 32.
Peak RSS was 1844 MiB Rust versus 2093 MiB Python. Approximate sampled
process GPU peaks were 1606 MiB Rust versus 2300/3726 MiB Python at batches
16/32. Rust CUDA-event spans (328.21/662.22 ms) include intermediate host
readbacks and idle time. There were 343 telemetry samples, no query errors,
observed temperatures 53.03–72.41 °C, and desktop GPU processes were present.
The manifest, measurements and telemetry hashes are
`6cd8953b3ce9d2bff2c9338d6d733ba588d95c493a241bee1c49f914c47a1c01`,
`2e505429377b740ffdda401af72c9a39ee360b5e9402c94af16fe8c5bca3b5ab`, and
`de249e1f70f9f773b989421f58f3fdf0da16ad234534d5c72aa07740a39fbcfb`.
Raw records remain in ignored `reports/phase4-wmt14-microbatch-api-*`.
These measurements remain nonexclusive and do not qualify the known-failing
cached CFG path or the WMT Metal gate.

A same-four-row cached-CFG diagnostic reproduced the full capture's row 2181
failure. At the first difference (level 5, position 6), Python chose code 32
with scores 6.35247755 versus 6.35247612 for code 1071; cached Rust chose
1071 with scores 6.35247517 versus 6.35247850. All compared level 0–5 logits
remain within the existing accumulated band, with one level-5 argmax crossing.
The captured trace SHA-256 is
`85155099da0fd67ea41a71223d2260ddb5f4dbba8cf52aeadc958e24918cb5dd`;
the comparison SHA-256 is
`5f13e04aac88d4640034691bb7de009991c7812e4cd864764cd9d3d84433bdb3`.
This does not waive the exact CFG gate; the ordinary cache constructor
continues to reject CFG.

A same-four-row trace for the other mismatch (row 2658) finds its first
argmax crossing at level 9, position 8. Python chooses code 1922 with
scores 11.90689564 versus 11.90689182 for code 1813; cached Rust chooses
1813 with scores 11.90689945 versus 11.90690517. Levels 0–8 have no argmax
differences in the batch. A later level-10 difference follows the changed
history. The pinned Python trace SHA-256 is
`07dcf16e7029f40c97b88d384ddecc0feb299ffdfe566bc372aadfb388a6c14f`;
the cached CUDA trace SHA-256 is
`d2544515e1d59ada5fcee67203864115302f9d4f927769bba7089c49cfc252e9`.
This traces inspected data and does not waive the exact CFG gate.

A matched uncached Rust CFG trace of the same four rows exactly matches the
Python trace's codes and raw IDs; cached Rust differs in two codes and one
raw ID. Cached and uncached level-zero logits are bit-identical, while level 1
logits already differ by up to `3.2902e-5` and retain identical argmaxes. At
the first changed choice (level 9, row 2658, position 8), uncached scores for
codes 1922 and 1813 are 11.90690041 and 11.90689754; cached scores are
11.90689945 and 11.90690517. This establishes cached-path arithmetic drift
before the choice but does not isolate a particular operator or justify a
change to the exact gate. Uncached trace SHA-256:
`7fd6ab8da2998e623faf8c6bebe6e8101c317bc39b31edebebaebd80a6c3af84`.
The diagnostic was built and run with the same artifact and pinned Python
trace as the cached comparison; targeted strict Clippy passed. Two initial
builds failed to find the unversioned cuDNN linker name; a symlink under the
ignored report directory to the existing versioned library resolved the third
build. Logs are retained under `reports/phase4-wmt14-uncached-cfg-*`.

## Cached/uncached internal shape diagnostic

On the same previously inspected four-row CFG batch, explicit level-one
internal traces compare cached target positions against the corresponding
uncached positions. The input and first layer-normalization output are
bit-identical; first Q/K/V projection values differ in 66,526 of 73,728
positions, maximum absolute difference `7.1526e-7`. The cached and uncached
internal captures preserve the respective previously observed codes and IDs.
Their SHA-256 values are
`43f18a47e689cfc148d9a9ef05de534932f8de2783a9bc0b9f5eb7c530491ef0` and
`a7af5fa4944abdca62ba3480c996a3d5ac41d06d1f4e4a96a5d8b499ca14013c`.

In separate opt-in probes, matching the uncached Q/K/V matrix shape made that
projection bit-identical, but attention still differed. Matching the attention
query shape made its output bit-identical; matching the output projection made
the first residual identical. Matching all first-block projection shapes made
the whole first block identical, but block 1 still differed in 21,639 of
24,576 values, maximum `3.8147e-6`. That final shape probe still differed from
Python in two codes and one raw ID on row 2658; SHA-256
`9a5e74fdea968b2825dc6a314588e47615c0eb1efe6b25579b89ebc2eccc4df8`.
It pads work and allocates filler, so no speed or memory improvement is claimed.
Only row 2658's original batch was tested, not the separate row 2181 failure or
the full corpus. These observations identify shape-sensitive arithmetic, not
a complete correction; the normal cache and CFG guard remain unchanged.

The full `tools/check.sh --cuda` passed, including 18 real-device operator
cases; log SHA-256
`0099e8428bc68490779527226518460dc9902d130041d463c6c5ced33d7746ec`.
After a nonsemantic Clippy argument-count allowance, feature-enabled strict
Clippy passed and all three artifact-backed CUDA cache regression tests passed;
the test log SHA-256 is
`610af19ba15930a81bbb5fc5a30e1e8a5838c30a37124299c9daf54b884b3cf0`.
The first lint attempt's argument-count error is retained in its v1 log.

## Prefix-state follow-up

On row 2658's original four-row batch, the explicitly captured cached prefix
first normalization, Q/K/V and attention outputs are bit-identical to the
prefix portion of the uncached level-one trace. The first attention residual
differs in 469,379 of 589,824 values, maximum `7.6294e-6`; second normalization
and feed-forward values also differ. After block zero, 562,372 prefix values
differ, maximum `0.003418`. These numbers include padded positions. Cached
and uncached traces retain the respective prior code and ID decisions. Their
SHA-256 values are
`9e2ace3aa1ca37fbcb1872a532b59f67eda6977113b46084a9b7fc144e55e291` and
`7f8e4a1f23f674283a41847d9eee00e723b5b287411af569cda08eee490d3c52`.
The first difference is at the output projection after identical attention
inputs; different full-sequence matrix shapes are consistent with the earlier
shape probes, but a causal kernel-level proof is not claimed.

As a bounded exploratory check, top-two scores in each of the previously
captured cached CFG traces for rows 2181 and 2658 each contain one margin
below `1e-5`, `1e-4` and `1e-3`, across 1,252 positions per four-row batch.
The minimum margins are `3.33786e-6` and `5.72205e-6` respectively. These
inspected examples do not establish a safe margin or detectability guarantee
for the full corpus; no fallback or gate change follows from this observation.

The follow-up instrumentation passed full `tools/check.sh --cuda` with 18
CUDA operator cases (log SHA-256
`6adacfc2a9baca2754ffbc3a8d87dc0651013f0435c12f477d21eabc125154b3`),
feature-enabled strict Clippy, and all three artifact-backed CUDA cache tests
(log SHA-256
`5e49d84f422d9eb384a098f120e669436d53c1131c38099803ec146521c66eae`).

## Explicit near-tie recomputation diagnostic

An opt-in diagnostic reads each guided cached head's scores and recomputes the
whole level without the cache if any top-two margin is at most `1e-4`. The
margin controls diagnostic recomputation; it is not a numerical tolerance or
an established bound on cache error. The normal cached CFG constructor remains
guarded. Invalid margins, malformed scores and nonfinite scores fail.

On the two previously inspected four-row batches, this diagnostic matches every
Python code and raw ID. It recomputed level 5 on row 2181's batch and level 9
on row 2658's batch. The respective trace SHA-256 values are
`32409f1819e792d01b0cdd1765a82e99f597b607041e7092f99621bdc0249251` and
`451fc9a1ce0fdf4f85c8b481938681567fd1d05d3fbcdf7c2b7d27fc3cd6169d`.
A separate 35-row smoke matches the retained Python codes and raw IDs; its
output SHA-256 is
`c62c44e40d35d4ca31441d72cd8601d627e2f9b4f4549fe631f4e1e4e8b546c8`.
These are reused inputs; full-corpus replay, scoring and performance remain
pending. The host score readback can be expensive, and no speed or memory
claim is made. The first strict Clippy run found a chunk-iteration style issue;
the corrected rerun passed. Full `tools/check.sh --cuda` passed 18 device
operator cases; log SHA-256
`253e386a1c025bfd6049eeef8f2711dab6ce7a086c281334fb89a0003ae59f62`.
The margin unit test and all three artifact-backed CUDA cache tests passed.

The source-bound complete 3,003-row CFG replay at `018216b` matched every
Python code and raw ID, including both previously failing rows. Its capture
and comparison SHA-256 values are
`f89220200f45e357fef9298849792ecaac3321e8deb2b4ca8a71989d70595915` and
`e1ea831fbfd216aa4d6b77ba9c88cf557d10640b3059de4093b6190d1b76d58a`;
the manifest SHA-256 is
`b921581dcc865573161903a2710dcd6f48047451d03b19672e79c7f9a0440360`.
The frozen full and held-out scorer passed with identical hypotheses; its
SHA-256 is `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.
Full-set BLEU was 22.9464, lowercase BLEU 23.8358, chrF 51.5160, and length
ratio 0.95867 on both sides. This reuses inspected data and does not establish
a bound for unseen inputs. The first decode attempt used a Python environment
without Transformers; the existing pinned tokenizer environment succeeded.
A source-bound alternating performance comparison is prepared; results remain
pending and the ordinary cached CFG guard remains in place.

The frozen scorer was independently replayed and reproduced byte-for-byte
with the same SHA-256. An alternating three-process comparison at `45d4fed`,
with five warmups and 30 trials per process, matched every benchmark code and
raw ID. Pooled median/p95 latency in milliseconds, peak process RSS, and
approximate sampled process GPU peaks were:

| Batch | Python median / p95 ms | Rust median / p95 ms | Python / Rust RSS MiB | Python / Rust sampled GPU MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 100.79 / 101.29 | 51.29 / 51.76 | 2093.10 / 1844.70 | 1364 / 1318 |
| 4 | 328.78 / 330.00 | 139.12 / 140.28 | 2093.25 / 1856.57 | 1538 / 1384 |
| 16 | 1235.62 / 1238.85 | 613.08 / 615.26 | 2092.75 / 1911.43 | 3072 / 3176 |

Rust sampled GPU usage is higher at batch 16 and therefore remains a visible
memory failure, not a full performance pass. These are approximately one-second
process samples, not native allocator peaks. All 319 telemetry samples had
zero query failures; observed temperature was 40.00–71.34 °C and other desktop
GPU processes were present. Process medians in milliseconds were Rust
51.13/51.35/51.33, 138.74/139.03/139.78, and 611.63/613.09/614.12; Python
100.59/100.77/100.97, 327.37/328.82/329.51, and 1231.51/1236.04/1237.70.
Manifest, measurement and telemetry SHA-256 values are
`4dc42d1f700618214f0808dbace680bb76be11ac993f7abb6a27208f3167c42c`,
`fea83cd63c856b8c0997ed04f7d59d45345c1fc71eab0136a6a2c1569775d1e6`, and
`bb790030008022109c9f759573632aa7d8db833ccb3cc2c1d65a57d7a93959c5`.
Telemetry is nonexclusive and does not establish allocator peaks or unseen
input correctness. CFG remains explicitly diagnostic and guarded.

## CFG recomputation microbatch diagnostic

An explicit CFG recomputation-only host-output method schedules external
batches up to 16 in groups of at most four. It rejects invalid batches,
prefixes and scales before execution, and rejects backends without the
explicit recomputation diagnostic. It does not qualify the heuristic or the
ordinary cached CFG path. Four artifact-backed CUDA tests pass, including
chunk equality for batch 1/4/5/16 and invalid inputs; log SHA-256
`76f89a52fc637ad9aa1df085b3f3fe43ba5d56725db3d2f8bfee2277e6827c87`.
A 35-row capture with external batches of 16 and a partial tail matches the
retained Python codes and raw IDs, SHA-256
`8e0b84495b6040e82241e5a0dda2de3a9c936ee50aab8122da994dc53e2fb737`.
Strict feature-enabled Clippy and the full `tools/check.sh --cuda` pass,
including 18 operator cases; full-check log SHA-256
`fac15f2ca0d66ebe04cde0d542c27caa1a50af546d8cf941951f8269abe74c0b`.
Source-bound full replay, quality and repeated memory/performance remain pending.
