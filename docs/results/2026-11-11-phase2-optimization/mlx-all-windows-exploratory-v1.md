# Apple control-window sensitivity across all timing cells

The earlier and quiet controls used the same native executable hashes, MLX
libraries, harness, Python environments, OS, compiler and workload declarations.
Each contains nine counterbalanced fresh-process triplets per workload. The
[analysis script](mlx-all-windows-exploratory-v1.py) checks all 270 raw control
report hashes against their original audits, checks source identities across
windows, and reproduces the 58 original controlled timing point ratios in each
window. The original audits separately pass all 20 host-memory comparisons in
both windows.

As a **post-hoc sensitivity analysis**, pool the 18 processes per runtime and
resample paired process indices within each window, using 10,000 fixed-seed
bootstrap draws. All **58 timing comparisons** have point estimates and
descriptive interval upper endpoints within the existing 1.05 median and 1.10
p95 bands. The tightest p95 comparison is B4 block versus previous Rust, with
upper endpoint **1.098714**. The largest median upper endpoint is **1.001473**
for BERT decode-with-specials-skipped versus previous Rust. All data from both
windows are retained.

This calculation was selected after observing the original failures. It is not
a predeclared acceptance audit, does not establish a 78/78 pass, and does not
erase the original **76/78** and quiet-window **77/78** control results.
Paired processes execute sequentially under potentially different machine
conditions; two windows do not establish stationarity. The per-cell 95%
percentile intervals are descriptive, not simultaneous guarantees across the
matrix. The B4 endpoint is close enough to the limit that a future independent
run could still fail.

## Existing tooling: offline 18-process replay

An [offline probe](mlx-gate18-offline-probe-v1.py) reconstructs a synthetic
18-process control from all nine processes in each retained window, without
changing their report bytes. It verifies each report against its original audit,
checks matching environment identities, and exercises the existing codec and
all four tokenizer summarizers. They retain 18 reports per runtime across all
29 scopes. The existing final audit then constructs all 78 cells and accepts
the synthetic control structure; deleting one declared codec report causes it
to reject the run. The command completed successfully on Linux using the pinned
reference environment:

```sh
.reference/venv/bin/python docs/results/2026-11-11-phase2-optimization/mlx-gate18-offline-probe-v1.py
```

This checks tooling compatibility and rejection behavior only. It combines
historical windows and synthetic metadata; it is not a fresh independent gate,
a prediction of the next result, or authorization to run one. The two original
failed controls remain unchanged. The two source archives have SHA-256
`3448f9bd9510bce353970802b57fbd42023855e4779798d4f58d08de4a3031d9` and
`411f5555a7949003be115239dc03f5b3e4060c5c75350b52b5cf46a19c30ac88`.

## Prospective option for review

If a new unchanged-candidate measurement is explicitly chosen, a more stable
*prospective* protocol could use one fresh three-repeat full matrix followed
unconditionally by one fresh **18-repeat all-workload control**, using the
existing counterbalanced runner and the unchanged 78-cell audit, workloads,
FP32 tolerances, p95/median/memory limits, output qualification and activity
monitoring. That is 315 benchmark processes in total (3 runtimes × 5 workloads
× 21 repeats), roughly 25 minutes based on the prior Mac collections. The
number, full scope and one-control stopping rule would be fixed before launch;
all failures and partial runs would remain visible, and no follow-up would be
selected based on its result. Reuse qualified trace/lifecycle evidence only if
source, executable and environment bindings are verified.

This is a proposal, **not authorization or a changed gate**. The previous plan
prohibited another unchanged-candidate retry, and the user authorized the one
separate quiet-window run already performed. A further run requires a new
decision. Even a future pass would be additional evidence alongside the two
retained failures; it would not retroactively make either prior control pass or
automatically close Phase 2. The alternative is to pursue a defensible runtime
change before any new gate. The compiled-GELU prototype was deferred after mixed
full-codec results, so no such change has yet been selected.

Reproduce the sensitivity analysis from the repository root with the pinned
reference environment (NumPy 2.5.4):

```sh
.reference/venv/bin/python docs/results/2026-11-11-phase2-optimization/mlx-all-windows-exploratory-v1.py
```

The calculation also reproduced from a clean source export with that same
environment; no GPU workload or acceptance policy was changed.
