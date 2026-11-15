# Exact GELU scalar reuse: isolated diagnostic

The MLX backend builds three identical scalar arrays for each exact-erf GELU
call. This isolated example compares that existing expression with reusing three
immutable arrays, keeping the same operator order and FP32 values. Unlike the
prior compiled-GELU microbenchmark, it includes fresh scalar construction in the
baseline timing. It does not change model code, use compilation, or introduce
shape-specific production behavior.

The current qualified Linux CUDA 78/78 gate remains applicable: this commits
only an unreferenced Apple diagnostic source and documentation. Before Metal,
verify the production source and existing qualified CUDA executable identities.
On Metal, run three fresh diagnostic processes, each with alternating order,
five warmups and 30 trials at B1/B2/B4 × 89 × 1024. Verify finite, bit-identical
full outputs before timing and preserve every result. Record binary/source
identities and observed activity. This is a microbenchmark, not acceptance or
trained parity. If there is no clear benefit, do not integrate; if promising,
prepare a separately reviewed bounded model/backend-owned design and validate
Linux first, then Metal correctness, trained parity, lifecycle, full codec and
the unchanged gates. Do not infer that an isolated speedup fixes the observed
scheduling tail.
