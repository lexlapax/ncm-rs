# MLX area resampling — isolated prototype

An isolated padded gather, masked reduction, and division was compared with
the current per-bin Metal implementation in three fresh processes per mode,
with five warmups and 30 trials for each of six shapes. Finite sampled outputs
passed the existing tolerance; maximum observed absolute difference was
4.77e-7. This is not trained-model qualification or an acceptance result.

Median evaluated time ratios for batch 1 at 89-to-1 and 89-to-2 were
**1.153 and 1.087** (regressions); 89-to-13 and 89-to-50 were **0.890 and
0.578**, and 13-to-89 was **0.495**. Measured MLX peak counters also rose:
for batch 4, 89-to-13, 341,804 to 585,090 bytes. A bounded hybrid candidate
retaining scalar handling for the smallest or oversized cases is prepared for
further real-Metal validation; no production area change has been made.

[Raw reports and prototype source](raw/mlx-area-poc-v1.tar.xz) SHA-256:
`5de7daea6511db1b028b4a0645286191d0fc82c726d4090f9047f66365e1a690`.

## Bounded hybrid follow-up

An isolated hybrid retains scalar bins for output lengths one and two or when
its padded gather would exceed four million elements, and uses gather/select/
reduction otherwise. It passed the existing Metal operator and codec-component
tests, expanded finite and nonfinite comparisons, and all 642 trained traces /
2,500 decisions against the unchanged oracle. No production area code changed.

Three matched processes per native binary compared this candidate with the
committed linear-gather implementation across the full codec. The candidate's
batch 1/2/4 encode median ratios were 0.830/0.817/0.485, but resident decoder
block ratios were also 0.716/0.693/0.585 even though that scope does not invoke
area resampling. Decode also improved broadly. These results are confounded by
process/runtime variation or indirect effects; they do not establish a causal
full-model area speedup. The candidate's batch-4 reconstruction MLX peak
counter rose from 155,858,012 to 193,539,880 bytes, while observed host HWM
was lower in this sample. The hybrid remains deferred pending a defensible
full-model benefit and resource tradeoff; no adaptive retry was run.

[Hybrid source, tests, qualified trace report, logs and full-model diagnostic](raw/mlx-area-candidate-v1.tar.xz)
have SHA-256 `a0429bfc578edc47be8e733cde2f26e284cfd55e605512c5c60877991731fee5`.
