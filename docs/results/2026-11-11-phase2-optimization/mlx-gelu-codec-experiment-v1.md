# Isolated full-codec compiled GELU experiment

The finite-checked [microbenchmark](mlx-gelu-compile-poc-v2.md) showed lower
warm B4 operation time but material first-use cost. The adjacent experimental
patch applies only in an isolated checkout. It retains the existing exact-erf
expression and passes the input and scalar arrays explicitly to MLX compile.
Only F32 shapes `[1|2|4, 89, 1024]` use it; other shapes use the original
path. A lazily created compiler is shared by backend clones and can specialize
only for those three shapes. This is an experiment, not an approved production
change. MLX compile's cache creation can panic on failure; cache retention,
thread/clone behavior, non-finite values and startup cost require review.
The zero-context patch must be applied with `git apply --unidiff-zero` only
after verifying the original codec SHA-256
`2283893a503fbf18de4a92024971cd4cac010a2032acde1f752c14fe39bdea4f`
and component-test SHA-256
`886d1ecf2fba7aedd2c5c04334590db0fd2fbf3186eeafab39913fb1e934aef9`.

The isolated Linux checkout passed `tools/check.sh --cuda`, including all 18
real CUDA operator cases, with a private umask and fresh target. Its codec
benchmark, trace and lifecycle executables are byte-identical to the previously
qualified Linux candidate, respectively SHA-256 `925363bce9e04204f27f4035fdc9a305aac40cfd77f5a02d5baffcfb860e20e0`,
`7dc4cb57ef9a3d0e0eabd496ccdeb35d6fbb1e579cda39278099862ae1701c85` and
`be60ca9b10ac0d5a3063b0548a25b8b978ac5775183611986ca923d5d14c80ed`.
Existing Linux quality and 78/78 gate evidence are retained on that binding.

The first Metal focused test failed to compile before running: the test's eager
reference annotated an `AsSliceError` as an MLX `Exception`. The separate
[v2 patch](mlx-gelu-codec-experiment-v2.patch) changes only that test error
type; the runtime patch is identical. The corrected focused test and full
`tools/check.sh --mlx` pass, including 18 real Metal operator cases. The trained
capture qualifies against the pinned CUDA reference: **642 traces, 2,500
decisions, zero ID disagreements, two near ties**. Its patched codec source
SHA-256 is `8e78d05b2edbe11815b14ad949fa0eb6cc0bfd479a71526f6f708634836dc9c4`;
the captured trace SHA-256 is
`27cce6ff04a079f7c4fb23bebc44906a4057dbb76cb4da78377f2c20662a4dcb`.
The [Metal checks, trace report and qualification](raw/mlx-gelu-codec-metal-check-v2.tar.xz)
have SHA-256 `37de7bb8019c0130fdee978955868c0da94d1c7416ed1e2cf3916fa77f02bc45`.
The large trace remains local and on the Mac. The
[Linux check and failed Metal build logs](raw/mlx-gelu-codec-checks-v1.tar.xz)
have SHA-256 `dfc03a3a99dc44c45e29a841c2d366b66035462455dae5aecc8b02d187d82e8b`.

## Full-codec diagnostic: mixed, deferred

The fixed three alternating process pairs completed with matching input/output
hashes in all codec scopes. All six source inventories are internally stable
and differ only in the intended MLX codec source. Against the current runtime,
B4 resident-block median ratios were **0.9157, 0.9919 and 0.9442** across the
three pairs. B2 block ratios were **1.1651, 1.1442 and 0.9617**, and B2
reconstruction ratios were **1.3278, 1.1469 and 0.9785**. Other scopes were
mixed. These three-process observations do not establish that compilation
caused those regressions or that B4 tail acceptance would pass.

Fresh-process first execution was **28.43–29.47 ms** current and **29.14–29.46
ms** compiled; the framework/system compilation cache had been warmed by
earlier experiments, so this is not a cold-compile bound. Final host HWM was
277.95–278.20 MB current and 278.51–278.87 MB compiled. Activity sampling
recorded 37 samples, no Chrome, one wallpaper appearance, and WindowServer
median/peak CPU 3.75%/4.6%; no thermal warning was recorded. These samples do
not establish exclusive GPU access.

The B4 microbenchmark improvement did not translate into a consistently useful
whole-codec result, while B2 and cache/startup tradeoffs remain unresolved.
Defer this prototype; do not integrate it or run an acceptance gate for it.
Do not infer that restricting an optimization to a benchmark-specific shape
would solve the underlying variability. The [raw paired reports, runner and
activity](raw/mlx-gelu-codec-diagnostic-v1.tar.xz) have SHA-256
`41b27693dc394e1abda6f7b88e51824ce9a0c7b762fddfc10905052bb5c72bf4`.

Validate Linux/CUDA first, including a build/source binding showing whether
the Apple-only change affects Linux binaries. On real Metal, first compare
supported and fallback shapes with the eager expression, including non-finite
inputs and backend clones, and qualify trained traces with unchanged budgets.
Then compare the full codec in three fresh paired processes against the same
runtime without the patch, reporting B1/B2/B4 block and codec scopes, first
execution, checked load, host memory and matching outputs. Record activity and
all outcomes. The diagnostic is not an acceptance gate. Do not integrate or run
a new full gate unless correctness, full-codec performance and the startup and
cache tradeoffs justify it; preserve any negative result.
