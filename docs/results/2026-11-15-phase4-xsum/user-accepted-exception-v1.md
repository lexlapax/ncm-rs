# XSum Metal acceptance for advancement to WMT14

On 2026-11-18, the user approved advancing to WMT14 with the 15
enumerated output differences in the retained XSum Metal top-1 run.
The decision applies to the 11,334-row batch-32 comparison and the
borrowed-upload capture independently verified to produce identical
Metal codes and raw IDs. The exact comparison remains recorded as a
failure. The existing criteria govern future runs and other profiles.

| Row | Intermediate code position | Python CUDA | Metal |
| ---: | ---: | ---: | ---: |
| 4937 | 8 | 3182 | 3690 |
| 4937 | 139 | 3793 | 3888 |
| 4937 | 185 | 614 | 3019 |
| 9813 | 129 | 2854 | 3206 |
| 10009 | 69 | 3758 | 3032 |
| 10009 | 98 | 2800 | 1056 |
| 10009 | 99 | 2692 | 1056 |
| 10009 | 136 | 2242 | 571 |
| 10015 | 170 | 2298 | 605 |

| Row | Raw ID position | Python CUDA | Metal |
| ---: | ---: | ---: | ---: |
| 4883 | 40 | 50256 | 3646 |
| 4937 | 34 | 50256 | 262 |
| 4937 | 42 | 257 | 531 |
| 4937 | 44 | 1941 | 737 |
| 4937 | 46 | 13 | 50256 |
| 4937 | 47 | 50256 | 13 |

Both raw ID rows have their first EOS at zero-based position 25.
All decoded texts and per-row ROUGE values match. Independent scoring
of the retained Metal outputs with pinned GPT-2 reproduced the Python
aggregate perplexity of `82.62335629362389` over 277,595 scored tokens.
[Full scoring replay](cuda-combined-numerics-v1.md#full-corpus-metal-text-and-rouge-replay).

The strict comparator reproduced its original failed receipt
`c6abd9084889fafc2b6eb804e2d7bbb0f5e37dc9d4af743c022fc3b40472a5d6`.
The candidate-to-Metal exact comparison receipt is
`c92c7367d40eb48a0304676b2f0a4f4003061e0c80a6e875d5b9db5294d99224`.
The independent perplexity replay receipt is
`12368660cb13a53e88c037cadc4e710dcd7517b6a379438cb5f2a576dfb79bfc`.
The [batch-size and state diagnostics](mlx-full-top1-v1.md#pinned-python-free-generation-across-batch-sizes)
remain scoped to their tested inputs.
