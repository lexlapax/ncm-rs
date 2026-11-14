# Strict-FP32 fused projection: Linux gate

User-approved candidate `928894b` passes the predeclared controlled Linux
acceptance audit: **78/78 comparisons**, with the pinned Python reference,
unchanged tolerances and unchanged performance limits. The initial 45-process
matrix passed **76/78**; BERT decode and skip-special process envelopes missed
the initial limit. An independent unconditional 135-process control covering
all workloads and scopes passed all 78. Nine fresh lifecycle processes completed
with identical output IDs. Initial and earlier failed gates remain retained.

Controlled median ratios versus accepted Rust `3307dff` were:

| Scope | Median ratio | p95 ratio | p95 95% interval upper |
| --- | ---: | ---: | ---: |
| Batch-1 resident decoder block | 0.75191 | 0.74432 | 0.74562 |
| Batch-2 resident decoder block | 0.73640 | 0.73589 | 0.74074 |
| Batch-4 resident decoder block | 0.72162 | 0.72140 | 0.72369 |
| TinyStories decode | 0.86225 | 0.81774 | 0.86514 |
| BERT decode | 1.00854 | 1.02214 | 1.03395 |
| BERT skip-special decode | 1.02338 | 1.02375 | 1.04083 |

The initial/control GPU monitors recorded 151/449 samples, each with median
utilization 3%, including benchmark work. The process monitor recorded nonzero
terminal activity. These samples do not prove exclusivity or explain individual
outliers. No clock or power policy changed. Fresh upstream-reference trained
qualification passes 642 traces and 2,500 decisions with zero ID disagreements;
see the [integration report](fused-integration-v1.md) for the check, tokenizer
probes, cross-stream regression, strict FP32 and workspace tradeoffs.

[Raw matrices, full control, plan, telemetry, lifecycle, qualification and audits](raw/fused-linux-gate-v1.tar.xz)
have SHA-256 `b5f0e8f017fae0a5ce4dd87bcb13b56e931be1d96b78e8636f507eb1884b4443`.
The controlled audit SHA-256
`cae2ab756615525620f4abe14d1ed1e90612f7e8331703f0e8a9964795c1235b`
reproduces byte-for-byte from a clean `928894b` export. All 120 native report
inventories and 180 shared Python report copies match clean candidate/baseline
exports. The benchmark's existing inventory omits the new backend module and
trait file; a separate manifest verifies both against that export, the compiler
depfile lists both, and the gate binary matches the isolated clean build.

Apple validation is next. The original Mac Tailscale entry went offline and
SSH to it timed out. A second online Mac-named entry presents the existing
trusted Mac SSH host key and reaches SSH authentication, but the local
1Password agent refused to sign. The user has been asked to unlock or approve
the agent. Mac checkout state and competing work must be rechecked before
pulling reviewed commits and running MLX. The Linux pass does not establish an
Apple pass or complete Phase 2.
