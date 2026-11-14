# Block signpost diagnostic plan

The existing focused trace exposed Metal intervals but did not align them with
block trials. The adjacent patch applies only to the isolated `ddde851` block
diagnostic, not production. The C shim emits one nonoverlapping Points of
Interest interval per block call and a graph-completion event, with only batch
and monotonically increasing sequence number. Initialization precedes model
loading. The Rust wrapper is diagnostic-only; its unsafe calls invoke the
fixed C signatures on the main benchmark thread after initialization.

Build with a separately compiled shim and isolated Cargo target, then perform
one bounded Metal System Trace with Points of Interest enabled. Verify the
executable/source identities, normal exit and qualified scope hashes; export
the marker and GPU tables and confirm the 108 block intervals and matching
graph events, sequence order, batch labels and measured duration alignment.
Do not infer GPU execution from wall time, or call a missing/unaligned marker a
success. If there is no slow trial, record that result and do not manufacture
an attribution. Trace overhead perturbs scheduling and is not acceptance data.

This is a diagnostic, not another full gate. Retain prior failures and limits;
do not publish the full system trace without reviewing system metadata.
