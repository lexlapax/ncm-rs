# Shape-independent exact GELU diagnostic

Pinned MLX supports [shape-independent compilation](https://github.com/ml-explore/mlx/blob/v0.32.2/docs/src/usage/compile.rst#shapeless-compilation).
The pure exact-erf expression has no shape-dependent branches or captures. This
isolated patch changes the prior toggle experiment to compile all rank-three
FP32 GELU inputs with `shapeless=true`; other ranks use the existing eager
expression. The same backend-owned compiler is shared by its clones, and all
arrays remain explicit inputs. This removes the prototype's three hard-coded
shapes, but does not by itself establish bounded GPU kernel caches or safe
compiler error handling. It is diagnostic code, not a production change.

First verify the qualified Linux sources and binary identities. In a separate
Metal checkout, apply the zero-context patch after confirming the original
backend SHA-256 `2283893a503fbf18de4a92024971cd4cac010a2032acde1f752c14fe39bdea4f`.
Run the focused real-Metal regression across varying rank-three shapes,
noncontiguous views, special FP32 values, clones and rank-one/two fallbacks.
If that passes, reuse the same-process model comparison in three fresh processes
for B1/B2/B4 reconstruction and block scopes. Record all data, first execution,
thermal and activity; do not treat it as a gate or attribute a speedup to one
cause. Preserve failures. The current `mlx-rs` compile constructor calls
`expect` when obtaining its compile cache: a production design must explicitly
resolve this panic behavior, evaluate startup and cache lifetime, and pass full
Linux-first, trained-model, resource and platform acceptance checks. Neither a
successful regression nor a benchmark alone authorizes integration.
