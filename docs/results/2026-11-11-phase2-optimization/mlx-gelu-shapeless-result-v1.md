# Shape-independent exact GELU diagnostic

The isolated [predeclared candidate](mlx-gelu-shapeless-plan-v1.md) uses MLX's
shape-independent compilation for all rank-three FP32 GELU inputs and the
existing eager expression for other ranks. The regression passed on real Metal
for changing shapes, noncontiguous views, finite values, signed zeros, infinities,
NaNs, backend clones and rank-one/two fallbacks. Command in the isolated checkout:

```sh
cargo test -p ncm-backend-mlx --features mlx --release --locked --offline --test gelu_shapeless -- --exact shapeless_gelu_matches_eager_across_shapes_views_and_clones
```

The isolated target was cloned from the earlier toggle target; `cargo clean -p
ncm-backend-mlx` removed zero files, and Cargo rebuilt the local MLX backend,
core, IO and model packages for the new worktree. The pinned third-party build
artifacts were reused. Linux-qualified production source and executable hashes
were checked before Metal. No production source was changed.

Three fresh processes alternated eager and shapeless compiled execution in the
same trained model. Initial finite outputs met component tolerances and IDs
matched; later IDs remained stable. Ratios below are compiled/eager; p95 is
nearest rank over 30 timings after five warmups. First compiled scope time is
for the entire scope, not an isolated compile cost.

| Process | Batch | Scope | Median ratio | p95 ratio | First compiled scope ms |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | reconstruct | 0.9923 | 0.9771 | 261.016 |
| 1 | 1 | resident_decoder_block | 0.9546 | 0.8222 | 0.747 |
| 1 | 2 | reconstruct | 0.9954 | 0.9719 | 6.973 |
| 1 | 2 | resident_decoder_block | 0.9155 | 0.8078 | 0.786 |
| 1 | 4 | reconstruct | 0.9942 | 0.9829 | 9.029 |
| 1 | 4 | resident_decoder_block | 0.9243 | 0.8891 | 1.020 |
| 2 | 1 | reconstruct | 0.9981 | 0.9976 | 4.938 |
| 2 | 1 | resident_decoder_block | 0.9422 | 0.9341 | 0.737 |
| 2 | 2 | reconstruct | 0.9972 | 0.9812 | 6.875 |
| 2 | 2 | resident_decoder_block | 0.9243 | 0.8130 | 0.721 |
| 2 | 4 | reconstruct | 0.9934 | 1.0006 | 8.454 |
| 2 | 4 | resident_decoder_block | 0.9428 | 0.8954 | 0.998 |
| 3 | 1 | reconstruct | 1.0010 | 0.9992 | 6.365 |
| 3 | 1 | resident_decoder_block | 0.9544 | 0.9693 | 0.717 |
| 3 | 2 | reconstruct | 0.9838 | 0.9964 | 5.463 |
| 3 | 2 | resident_decoder_block | 0.9261 | 1.0358 | 0.752 |
| 3 | 4 | reconstruct | 0.9904 | 1.0169 | 8.491 |
| 3 | 4 | resident_decoder_block | 0.9505 | 0.9676 | 0.888 |

Resident-block medians improved in all nine comparisons; B4 ratios were
0.9243/0.9428/0.9505. Reconstruction medians ranged 0.9838–1.0010. First
compiled B1 reconstruction was 261.016 ms in the first process and 4.938/6.365
ms in the others. The earlier focused regression and system caches can affect
these observations; this is not a controlled cold-start measurement. Thirteen
activity samples saw WindowServer up to 32.4% CPU and cannot establish GPU
exclusivity. No thermal warning was recorded. This is not an acceptance gate.

The candidate removes hard-coded benchmark shapes but remains experimental.
The pinned Rust compiler constructor can panic on cache acquisition, and
startup, retained cache resources, broader model parity, full-codec behavior
and both runtime gates remain unresolved. Do not integrate or present it as a
complete fix without addressing those requirements. The [earlier separate-process
results](mlx-gelu-codec-experiment-v1.md) and all failed gates remain retained.

The patched backend SHA-256 is `6dd2a442fa85168f05b80d736b77eed2ad52ff8ebbccccbeea2ea4e151f72fd9`,
regression source is `6e47c83930b5ce202b48d181fe4e947192b315db585c0c42b1f969c98a952d7f`,
and diagnostic binary is `a83a4a17aeb1d4004ae98a228725ca2ea607f5e568a77867d1cac9fdec631bba`.
The [three reports, commands, activity and runner](raw/mlx-gelu-shapeless-diagnostic-v1.tar.xz)
have SHA-256 `cc815052a39273ae8cddb3f95f02027d031dd20214fd5c04a1e81b142e62008a`.
