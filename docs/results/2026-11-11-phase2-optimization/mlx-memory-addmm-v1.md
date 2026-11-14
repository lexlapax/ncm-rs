# Bounded MLX memory observation for projection candidate

Diagnostic branch `d24a4c3`, with runtime matching `a323b16`, ran 128 batch-4
reconstructions in each of three model lifetimes. All 384 output-ID hashes
matched the existing qualified workload. After each model drop, active MLX
bytes were **0**, cached MLX bytes were **164,001,728**, and host RSS was
**275,890,176 / 307,527,680 / 307,609,600 bytes** across the three cycles.
The MLX active-allocation peak was **155,557,200 bytes**. At requests 1, 8, 32,
64 and 128, active and cached MLX allocation snapshots remained within a narrow
range after warmup; the report retains every sample. Cache was cleared once
before the observation, never between cycles.

This finite observation is not proof of an infinite bound or a timing result.
Host RSS increased across cycles, and its cause remains unestablished. MLX
allocator counters are not separate VRAM or disjoint from host RSS.

[Raw report, build log, dependency file and hashes](raw/mlx-memory-addmm-v1.tar.xz)
have SHA-256 `af4f6961695404e20bfa46d974e1ce651ba18d07a839f688057eca0f4f4628be`.
The report's source hashes match the branch. The measured executable SHA-256
is `4064de9988c676d69a16e277ef59889557b573192e9ce004827f46a798e61575`.
This evidence does not close the separate TinyStories host-memory gate.
