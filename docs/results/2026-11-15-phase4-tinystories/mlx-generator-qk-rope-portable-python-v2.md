# Guarded Q/K rotary: repeated Python MLX comparison

The exact isolated [guarded, width-aware candidate](mlx-generator-qk-rope-portable-screen-v2.md) was compared with the qualified same-device Python MLX adaptation. For each batch, three alternating fresh processes per runtime ran five warmups and 30 measured requests. Both measure generation and readback; Rust additionally drops outputs and includes a GPU barrier. Its process-local cache limits were 512 MiB at batches 1/4 and 4,096 MiB at batches 32/128; Python used its default. The streams differ.

| Batch | Python / Rust median ms | Python / Rust pooled p95 ms | Python / Rust maximum process RSS MiB | Python / Rust maximum MLX peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 107.44 / 94.09 | 107.93 / 94.71 | 1046.75 / 1024.78 | 609.47 / 607.39 |
| 4 | 169.96 / 143.38 | 171.48 / 144.34 | 1047.11 / 1026.41 | 975.69 / 971.21 |
| 32 | 690.63 / 502.69 | 699.25 / 506.67 | 1046.80 / 1022.80 | 2528.94 / 2420.17 |
| 128 | 2572.83 / 1992.10 | 13657.40 / 2237.22 | 1045.88 / 1022.33 | 2539.28 / 2475.29 |

Rust was faster and lower in the listed memory counters in every measured batch. The batch-1 and batch-4 MLX peak margins were only 2.08 and 4.48 MiB. At batch 128, Python process medians were 2288.41, 2941.35 and 2572.83 ms; Rust's were 1926.75, 1992.10 and 2010.78 ms. Eight Python trials exceeded four seconds, with a 17784.83 ms maximum; no Rust trial did, with maximum 2370.57 ms. This run and the [prior unguarded run](mlx-generator-qk-rope-python-b128-v1.md) do not identify the tail cause or erase the [earlier failed Rust result](mlx-generator-compiled-gelu-python-performance-v1.md).

The candidate's own 16-row fixed-code summary qualified capture, and watched source and inputs remained stable. CPU activity sampling reported no errors; GPU activity was not measured and no exclusive reservation was available. RSS includes startup/loading; MLX active, peak and retained/cache counters have different scope, and the configured cache limit is not a hard instantaneous allocation bound. At batch 128 maximum Python/Rust active counters were 479.29/477.63 MiB and maximum retained/cache counters were 24597.57/4090.98 MiB. Neither this result nor passing the full hardware suites resolves the documented upstream cross-runtime numerical exceptions or qualifies other text profiles.

Copied ignored evidence is `reports/phase4-qk-rope-portable-python-v2/`: summary SHA-256 `a24b0d6d0c3e31fc69c15c25c74e20730f8630a3e9716932d8d0dbc73c7f62f4`, manifest `c2d143c96c23d3cdfe5edf4345804ffe2d77f9cbd280bf4fcc9cc8ab5cb65188`, CPU activity `d5868728de382d4fb3345c09871359ca458a6a5b9cb80d430e36f029edd6af4b`. All matched the Mac originals.
