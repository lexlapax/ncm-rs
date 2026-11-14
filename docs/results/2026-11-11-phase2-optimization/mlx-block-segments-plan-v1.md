# Metal block tail attribution — diagnostic plan

The current Apple initial matrix passes 78/78 but its predeclared control is
76/78 because batch-4 block p95 intervals remain open. One of nine candidate
processes accounts for a pronounced late tail; the cause is not known. Rather
than change the model or repeat the acceptance gate, this separate diagnostic
copies the existing benchmark workload and records, for each resident-block
trial, host graph-construction time and the subsequent synchronous evaluation
interval. Three fresh processes will retain the original prior scope sequence,
five warmups and 30 trials per batch. Activity and thermal observations will be
retained. Output identities must match the existing fixture, and source and
binary identities will be recorded.

Timers and bookkeeping perturb the code: the evaluation interval includes host
submission, synchronization and waiting, and is not a GPU-kernel-only time.
The diagnostic does not change or replace the acceptance matrix, prove a cause,
or authorize changes to GPU policies. Compare paired segments with the same
instrumented trial's wall time to locate the observed tail before selecting any
further optimization.

`cargo fmt --all -- --check` and Linux `cargo check --workspace --locked` passed.
The diagnostic example is feature-gated and was not compiled on Linux. It is
not yet compiled or run on Metal. No production inference source is changed.
