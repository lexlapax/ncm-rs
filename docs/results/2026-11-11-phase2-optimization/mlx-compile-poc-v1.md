# Pure MLX compilation — isolated experiment deferred

An isolated pure layer-normalization graph passed all arrays, including epsilon,
as explicit inputs to pinned MLX compilation; no arrays were captured. Three
fresh processes each tested three shapes with five warmups and 30 paired trials.
Sampled outputs matched the uncompiled graph exactly. This is a microbenchmark,
not trained-model qualification or a production implementation.

Median evaluated compiled/plain ratios were **0.922, 0.881, and 0.923** for
batch/length/channels 1/89/64, 1/89/256, and 4/89/256. The absolute warm savings
were roughly 0.03–0.04 ms per invocation. First compiled-call times ranged from
about 0.6 ms to **509 ms**, with the first process also taking 439 ms for the
second shape; later processes were faster. Persistent framework/system caches
were not cleared, so these are observed startup costs, not a claim of controlled
cold compilation. Sampled first-call MLX peak counters reached 0.13, 0.39 and
1.51 MB by shape; allocator/cache counters are not host RSS.

The experiment does not establish full-model speedup, bounded specialization
or cache lifetime, trained parity, or a practical startup break-even. It is
deferred rather than integrated. [Raw reports, binary identities, plan and
source](raw/mlx-compile-poc-v1.tar.xz) SHA-256:
`05e9185a8504d87400d0c59c289a080f257a32037f4d6461c18ed851e28ea5da`.
