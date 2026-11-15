# System-wide Metal trace: captured block outlier

After the earlier [failed save](mlx-system-activity-failed-v1.md), the user
confirmed the Mac was available for one short Metal trace. The corrected runner
recorded for 35 seconds, allowed finalization to complete, and both recorder
and marked target exited zero. This is an instrumented diagnostic, not an
acceptance run. Its target executable SHA-256 was
`6e3c68c9beda39daab01d43f81b6b923319c52ba3377bd9b4e5605fbdb2796da`;
manifest, workload, source inventory, and all twelve input/output ID hash pairs
match the [earlier marked diagnostic](mlx-block-signposts-v1.md).

Thirty measured batch-4 block calls had a 0.815834 ms median. Using the same
exploratory 1.2×-median threshold, two were slow: 0.987292 and 1.617167 ms.
For the larger outlier and three near-median trials (0.813292–0.818583 ms):

| Measurement | Large outlier | Three near-median trials |
|---|---:|---:|
| Marked wall interval | 1.616834 ms | 0.813000–0.818250 ms |
| Target main thread running | 0.156250 ms | 0.151833–0.174417 ms |
| Target main thread blocked | 1.449542 ms | 0.631958–0.653833 ms |
| Target main thread runnable | 0.011042 ms | 0.010958–0.012458 ms |
| Target GPU interval | 1.063041 ms | 0.328458–0.347417 ms |
| Other-process GPU interval union during marker | 0.847667 ms (4 records) | 0 ms |

The main-thread state intervals cover each entire marker. In the outlier,
other-process GPU intervals also overlap 0.758333 ms of the target GPU interval
(three records); the union from marker start to the target GPU start is
0.089334 ms (three records). Those intervals are reported trace intervals,
potentially spanning multiple GPU channels; their overlap alone does not prove
exclusive occupancy or that the other work caused the slower target. Encoding
ended about 0.14 ms after the marker in both outlier and near-median cases.
The smaller slow trial had no other-process GPU interval overlapping its marker.
The comparison is selected after observing this one trace; it does not explain
all tails or establish why either prior Apple gate failed.

The private system-wide trace and exported tables remain on the Mac at
`.reference/phase2-mlx-system-activity-v2/`; they contain other process metadata.
The [reviewed aggregate, target-only report, runner metadata and analysis scripts](raw/mlx-system-activity-v2-aggregate.tar.xz)
have SHA-256 `3e034ee3e719b2eacc92e2673b58051fbe4e6b723afeba7c29bcf3790bbb6415`. Analysis scripts require the private exported
marker, GPU, application and thread-state tables; the reported aggregate
contains no unrelated process identities. The earlier failed recording and all
previous acceptance results remain unchanged.
