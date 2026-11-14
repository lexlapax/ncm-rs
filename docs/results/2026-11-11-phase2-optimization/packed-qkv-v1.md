# Packed Q/K/V layout — provisional Linux candidate

The decoder now reshapes and permutes the Q/K/V projection into one packed
`[3, batch, heads, length, head_width]` layout, then takes the three slices.
On Candle CUDA the reshape of the permuted tensor materializes the layout once;
the slices are contiguous views. Q/K normalization, bidirectional attention,
trace names and arithmetic are unchanged. Lazy MLX behavior has not yet been
measured. A regression compares B4 decoder output with four individual calls
using different inputs; CPU and real CUDA component tests pass.

Three matched CUDA benchmark processes per binary used the unchanged codec
workload, CPU 2, five warmups and 30 trials, rotating previous/candidate order.
The previous binary was the saved finite-scan candidate from `a6290db`, run from
an isolated archive of that source; the BERT change does not affect this codec
binary. Median of the three process medians, milliseconds:

| Batch / scope | Previous | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| B1 resident block | 0.192656 | 0.159378 | 0.8273 |
| B2 resident block | 0.235454 | 0.209444 | 0.8895 |
| B4 resident block | 0.316901 | 0.288169 | 0.9093 |
| B1 decode | 1.218733 | 1.195218 | 0.9807 |
| B2 decode | 1.593275 | 1.565405 | 0.9825 |
| B4 decode | 2.114520 | 2.084262 | 0.9857 |

B1 process medians ranged 0.179–0.200 ms before and 0.158–0.176 ms after;
the known timing clusters remain, so this small diagnostic is not an acceptance
interval. All twelve batch/scope output-ID hashes matched. Trained CUDA
qualification passed **642 traces / 2,500 decisions**, zero ID disagreements,
two near ties, and produced the same native trace SHA-256
`d180a52c011c8af15812d10941de8574b1671c2cd23de67e9abc828f7c5ecba4` as
before. CPU tests and Clippy, and the focused real-CUDA component tests,
including batch independence, passed.

The new packed temporary has a memory cost. In the matched benchmark, resident
block CUDA pool **used peak** increased by 182,272 / 364,544 / 729,088 bytes
for B1/B2/B4. Pool reserved peak at B2 rose from 64 to 96 MiB in all three
processes; both reached 96 MiB at B4. These are allocator pool counters, not
host RSS or a physical memory total. A separate candidate-only ten-cycle,
144-request full-model lifetime capture passed and returned live pool usage to
zero after every model drop; its first B2 request reserved 64 MiB and B4 96 MiB,
matching an earlier historical capture. That older run is not a new matched
comparison and does not erase the benchmark's B2 reservation observation.
The three benchmark process host HWM values overlapped at roughly 604–607 MB;
this is not a definitive memory comparison.

### Offline pool-counter comparison

The isolated benchmark's B2 reservation first rose from 64 to 96 MiB in the
**reconstruct** scope and remained at 96 MiB for the following resident-block
scope. The block counter therefore does not by itself establish a new reservation
during that block. In the later combined matrix, accepted Rust and the candidate
had identical reserved pool peaks for every corresponding scope: 64 MiB for B1,
64 or 96 MiB for B2, and 96 MiB for B4, in all three process pairs.

In that combined matrix, the peak *used* pool bytes for reconstruction were
63,779,664 / 62,893,984 (B1), 67,973,208 / 66,176,168 (B2), and 76,360,296 /
72,740,536 (B4), accepted / candidate. Decode and resident-block used peaks
were higher in the candidate by 207,952 / 390,224 / 754,768 bytes for B1/B2/B4.
These are aggregate, scope-ordered pool counters for the combined candidate;
they do not isolate Q/K/V as the sole cause or equate reserved bytes with RSS.
The candidate-only ten-drop capture above remains the lifecycle evidence.

The [offline audit](raw/packed-qkv-pool-audit-v1.py) verifies both archive hashes,
the combined candidate binary hash and identical per-role counters across the
three retained processes before printing [its results](raw/packed-qkv-pool-audit-v1.json).
Script SHA-256 `93ca8fe75d3192b7d6a3ea886f825854761cdfc1f4b6190749191da7d4c4e8ea`;
result SHA-256 `b5da908746e51e74c34fc09f3aff5662bd6dd0a0e3af2ccf8783a34b04192362`.
It performs no GPU work and does not change the acceptance result.

[Raw reports, qualification, focused CUDA log, lifetime capture and binary
hashes](raw/packed-qkv-v1.tar.xz) SHA-256:
`cfd2de5b6f10989d587d39cfa07eefa3a93e3dbb6cc2735cddb34dc9111c695a`.
The lifetime capture SHA-256 is
`f3da5b7f0ac857507a72fb7e70f925aedfe54960058d335f337822a8abf85d34`.
This is not a full 78-cell audit, a Python performance comparison or an MLX
qualification. The prior 76/78 gate belongs to the previous candidate. Full
Linux validation and resource acceptance precede any Apple run.
