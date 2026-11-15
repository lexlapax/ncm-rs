# TinyStories Metal batch-128 tail diagnostic

A targeted two-pair follow-up to the [three-process lifecycle comparison](mlx-lifecycle-performance-v3.md) alternated fresh processes in Rust/Python, then Python/Rust order. Each process used five warmups and 30 synchronized generation/readback trials at batch 128. The Rust process included output cleanup and a GPU barrier inside each timed request; both used the default cache policy for this batch. The source-bound runner is `tools/bench/run_phase4_mlx_tail_probe.py`; raw output is retained in ignored `reports/phase4-tinystories-mlx-tail-probe-v1/`. Its manifest records source and artifact hashes, commands, and process time brackets. All watched hashes remained stable and telemetry reported no errors.

| Process | Median ms | p95 ms | Maximum ms | Trials over 4 s |
| --- | ---: | ---: | ---: | ---: |
| Rust 0 | 2030.55 | 2066.86 | 2079.69 | 0 |
| Python 0 | 2187.16 | 2255.26 | 2259.21 | 0 |
| Python 1 | 2191.39 | 2269.24 | 2279.83 | 0 |
| Rust 1 | 2040.52 | 2062.77 | 2065.48 | 0 |

The earlier lifecycle run's pooled batch-128 p95 was 7016.875 ms Rust versus 2658.448 ms Python, with eight Rust trials above four seconds. That observation remains a visible exception; this shorter follow-up does not replace it or identify its cause. The follow-up had 145 CPU activity samples at approximately two-second intervals; WindowServer appeared throughout, and corespotlightd reached 152.7% CPU in two samples. GPU activity was not measured, and no exclusive reservation or clock/power change was made. These observations cannot establish that desktop activity caused either outcome. Python is the qualified same-device MLX adaptation, not original upstream PyTorch, and the policy is scoped to the dedicated benchmark process. The probe does not establish long-lived concurrent service behavior.

The manifest SHA-256 is `a1c4382556364c122bd35df5d980f4032a8ffe726496406afadd5939095ea9a9` and activity SHA-256 is `4c2fd45bac7bd3f5773c410a7f19ba975139cdb30acdcd7961405c0506ae8365`. Individual report hashes are Rust 0 `01d50794c47aca20af90d9f713ebd592644897bd69360aa3005af1c5c185ce30`, Python 0 `69ea56979d99953df8e26a42c0dd46a7ffd1e72a72ef602d73cadb97509b4b87`, Python 1 `46bb771e958861ce07b551666c3924906aea327c4e9770d3f1005252b8e8e5ed`, and Rust 1 `9e5cae99b2d16a581a4ce34e424ebfb70f6a3b1f79544ec2eb3a51972194881b`. The p95 values use the nearest-rank 29th of 30 trials per process.
