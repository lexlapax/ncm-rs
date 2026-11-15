# Same-process compiled GELU comparison

The earlier full-codec compiled-GELU experiment showed inconsistent paired
process results, including large shifts in encoder scopes that do not directly
call GELU. This isolated diagnostic alternates eager and the identical earlier
exact-erf compiled expression within one loaded model and one process. The
adjacent patch adds a **diagnostic-only** shared toggle, never intended for
production. It retains the earlier prototype's B1/B2/B4 shape restriction to
compare that same candidate; this is not a recommendation for a shape-specific
production optimization. It does not change shared model math or the upstream
formula.

Use the existing Linux-qualified source and binaries first; this commit changes
only isolated diagnostic files. Apply the zero-context patch with `git apply --unidiff-zero` in a separate
Mac checkout after
verifying the original codec source hash
`2283893a503fbf18de4a92024971cd4cac010a2032acde1f752c14fe39bdea4f`.
Run three fresh processes, each comparing all B1/B2/B4 reconstruction and
resident-block scopes in alternating order, after five warmups with 30 trials
per mode. The same retained latent, model, backend and process are used for both
modes. Initial finite floating outputs must agree within the existing component
budget and output IDs must match; subsequent IDs must remain stable. Record the
first compiled invocation, every timing and activity. The first-use observation
is not a cold system-cache measurement. Never drop an unfavorable scope or trial.

This is a focused diagnostic, not an acceptance gate. Same-process alternation
reduces between-process drift but introduces shared-cache and interleaving
influences; it does not replace the earlier full-codec results, production parity,
startup/lifetime review or the 78-cell gates. Integrate nothing without a
consistent useful result and a general, bounded design preserving upstream
semantics. No additional gate is authorized by this plan.
