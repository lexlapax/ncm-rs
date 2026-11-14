# Instrumented MLX block timing attribution

The current Apple initial audit is 78/78, but its predeclared all-workload
control remains 76/78 because batch-4 block p95 intervals exceed the locked
limit against both comparators. To investigate without repeating that gate,
diagnostic branch `ddde851` ran three fresh processes with the same preceding
codec scope order, five warmups and 30 measured trials per scope. It added
host timers around block graph construction and the following synchronous
evaluation. This diagnostic changes measurement overhead and is not an
acceptance run.

For offline inspection, using wall time greater than 1.2 times the
same process's median, one of 90 batch-4 block trials was an outlier. Its wall,
graph-construction and evaluation intervals were **2.3082 / 0.2689 / 2.0374
ms**, versus that process's medians **1.1599 / 0.0673 / 1.0963 ms**. The
other two processes had no trial above that threshold; their wall medians were
1.1259 and 1.1259 ms. Both instrumented intervals rose in the observed
outlier. The evaluation interval includes submission, synchronization and
waiting, so it is not a GPU-kernel-only measurement and does not identify the
cause. One outlier is insufficient to generalize the gate's distribution.

All three reports' scope input/output ID hashes match the qualified gate; the
resident-block scope itself emits no IDs. Nineteen activity samples showed no
Chrome, with WindowServer median/peak CPU 0.0%/5.0%; no thermal warning was
reported before or after. These samples do not establish exclusive GPU access.
The [raw reports, build log, compiler dependency file and runner](raw/mlx-block-segments-v1.tar.xz)
have SHA-256 `c9511a3cd33f6ee3c9f0552924f2d8709cad05ab9152bd2606d0c3346a3d1dfa`.
All three report source inventories match the diagnostic branch. The executable
SHA-256 is `3fb22a460b7c31af1c1626b2ccfdf0cbed015d49e71f7571c25bc96ac4237acc`.
No clock, power, tolerance, workload or production inference change was made.
