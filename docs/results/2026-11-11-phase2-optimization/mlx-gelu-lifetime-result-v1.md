# Shape-independent GELU: bounded lifetime result

The [planned observation](mlx-gelu-lifetime-plan-v1.md) ran one eager process
followed by one compiled process on real Metal. Each created and dropped three
backends/models and ran 16 trained reconstructions per lifetime, cycling batches
1–5. The artifact manifest and all output-ID hashes matched across modes. These
are 48 requests per mode, not trained numerical qualification or an acceptance
benchmark. The Linux-qualified source and codec executable identities were
verified first; the isolated Mac target cloned prior build artifacts and rebuilt
local workspace packages. No production runtime changed.

| Mode | First reconstruction in cycles 1 / 2 / 3 (ms) | MLX active after each drop (bytes) | MLX cache after drops (bytes) | Host RSS after drops (bytes) |
|---|---|---|---|---|
| Eager | 84.884 / 6.979 / 7.105 | 0 / 0 / 0 | 353,487,528 / 353,569,500 / 353,569,500 | 276,561,920 / 308,494,336 / 308,969,472 |
| Compiled | 266.292 / 7.074 / 6.694 | 0 / 0 / 0 | 343,733,376 / 343,733,376 / 343,733,376 | 277,233,664 / 309,542,912 / 309,985,280 |

The first compiled request was substantially slower in this observation; it
includes all reconstruction work and is not a measured compile-only duration.
Processes ran sequentially, eager first; system/framework caches and activity
were not controlled, so this is not a controlled cold-start comparison. The
later first requests were comparable. Active MLX allocation returned to zero
after all drops and allocator cache stabilized over these three lifetimes;
host RSS rose in both modes. These counters are process observations, not a
general bound or a measurement of all compiler/driver memory. Six activity
samples recorded WindowServer up to 31.7% CPU; they cannot establish exclusive
GPU access. No thermal warning was reported.

The pinned Rust binding's [built-in GELU](https://github.com/oxiglade/mlx-rs/blob/adf439bf5361d8a02955de7aaf651b0c45900945/mlx-rs/src/nn/activation.rs) also constructs a compiled function on each call; its [compiled-state drop](https://github.com/oxiglade/mlx-rs/blob/adf439bf5361d8a02955de7aaf651b0c45900945/mlx-rs/src/transforms/compile/mod.rs) erases that function's compile-cache entry. It uses the same exact GELU formula but a different FP32 operation order. Source inspection does not quantify its startup or rule out lower-level GPU shader caching, and no helper substitution was measured. It is not evidence of an ahead-of-time, reusable alternative.

The repeated large first-use observation is a material startup concern. Together
with the pinned binding's fallible cache acquisition implemented as `expect`, it
precludes integrating this prototype as-is. Do not move the cost to an
unreported load or later request, silently change error behavior, or use this
finite observation to waive the unchanged full-model and platform gates.

Patched backend SHA-256: `6dd2a442fa85168f05b80d736b77eed2ad52ff8ebbccccbeea2ea4e151f72fd9`;
diagnostic source: `0216d12f24c93b36d9ccfb461eb67b6da7bfe417d80b2252cbc57caf075f6917`; binary:
`ec0992902701ef4a7d98f0f140bd3c884993b15540b117da89210a0f44cafb84`.
The [reports, commands, sampled activity and runner](raw/mlx-gelu-lifetime-diagnostic-v1.tar.xz)
have SHA-256 `d162052821133bab2ee76fa29435477dc2b2a4bbc99880fc8dd3954b25b38e51`.
