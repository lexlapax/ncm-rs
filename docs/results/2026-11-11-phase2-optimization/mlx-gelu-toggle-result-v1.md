# Compiled exact GELU: same-process comparison

The predeclared [diagnostic](mlx-gelu-toggle-plan-v1.md) alternated the eager
and previously prototyped compiled exact-erf expression within the same loaded
model and process. Linux-qualified production sources and executable identities
were checked first. Three fresh Metal processes completed; initial finite full
outputs met the existing component tolerance and IDs matched, and IDs remained
stable during timing. There were five warmups and 30 timings per mode. Ratios
are compiled/eager, with nearest-rank p95; lower is faster.

| Process | Batch | Scope | Median ratio | p95 ratio | First compiled scope ms |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | reconstruct | 1.0060 | 0.9943 | 212.978 |
| 1 | 1 | resident_decoder_block | 0.9445 | 1.1143 | 0.744 |
| 1 | 2 | reconstruct | 0.9899 | 0.9857 | 7.076 |
| 1 | 2 | resident_decoder_block | 0.9369 | 0.9400 | 0.780 |
| 1 | 4 | reconstruct | 1.0088 | 1.0606 | 8.745 |
| 1 | 4 | resident_decoder_block | 0.9380 | 0.9061 | 1.062 |
| 2 | 1 | reconstruct | 0.9929 | 0.9843 | 6.289 |
| 2 | 1 | resident_decoder_block | 0.9466 | 1.1354 | 0.757 |
| 2 | 2 | reconstruct | 0.9988 | 1.0019 | 7.117 |
| 2 | 2 | resident_decoder_block | 0.9198 | 0.9619 | 0.769 |
| 2 | 4 | reconstruct | 0.9933 | 1.0027 | 8.579 |
| 2 | 4 | resident_decoder_block | 0.9448 | 1.0156 | 0.962 |
| 3 | 1 | reconstruct | 0.9992 | 0.9920 | 6.362 |
| 3 | 1 | resident_decoder_block | 0.9471 | 0.8915 | 0.742 |
| 3 | 2 | reconstruct | 0.9978 | 0.9910 | 7.134 |
| 3 | 2 | resident_decoder_block | 0.9183 | 0.9169 | 0.800 |
| 3 | 4 | reconstruct | 0.9973 | 1.0555 | 8.504 |
| 3 | 4 | resident_decoder_block | 0.9364 | 0.8742 | 0.953 |

Resident-block medians improve in all nine comparisons, including B4 ratios
0.9380/0.9448/0.9364; reconstruction medians range 0.9899–1.0088 and p95
ratios vary. This strengthens the case that compiled GELU can improve the warm
block itself, but does not show a material end-to-end improvement or prove a
passing Apple gate. Unlike the prior separate-process measurements, B2 block
did not regress in these same-process samples. Shared caches, interleaving and
scope order affect this diagnostic, and these three processes do not erase the
prior mixed full-codec results.

The first compiled B1 reconstruction took 212.978 ms in the first process,
versus 6.289/6.362 ms in the next two. This includes the entire reconstruction
and potentially compilation; framework/system caches were uncontrolled, so it
is not an isolated compile cost or a controlled cold-start bound. Thirteen
activity samples saw WindowServer up to 31.9% CPU; they cannot establish GPU
exclusivity. No thermal warning was recorded. The prototype is not integrated:
it still restricts shapes, has unresolved cache lifetime and compile-error
behavior, and requires a general bounded design, full-codec quality and
performance evidence, lifecycle checks and both unchanged acceptance gates.

The diagnostic binary SHA-256 is `3e7867eb765fa7cd58ebb4c1aaae187871c98cd654abfe4f83fc4656e530d66a`; the example
source is `1cea0e33b37592f3ba310fc319f1e1ffff72e4d8b1241f81583c04fb087c23d8` and the isolated patched backend is
`00026cd9eada9811f7cb089a19fa35aa6ad73e771c4850d96298481aa89be9be`.
The artifact manifest SHA-256 `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832`
matches the earlier codec diagnostic. The [reports, commands, activity and runner](raw/mlx-gelu-toggle-diagnostic-v1.tar.xz)
have SHA-256 `f0fcce98c33a0882f963bfbc26f77fed98ed9563ecaaed6c92be2afa66c5c39c`. No production code, pins or tolerances changed.
