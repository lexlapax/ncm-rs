# Apple 18-process gate and resource follow-up

On 2026-11-15 UTC the Apple matrix ran 3 fresh processes per runtime and
workload, followed unconditionally by the predeclared 18-process control.
The initial audit failed one of 78 cells: B4 resident decoder block p95 versus
prior Rust was 1.152595 (limit 1.10). The independently run control passed all
78 cells and quality: its B4 block p95 ratio was 0.951541 against prior Rust,
with paired-process 95% interval [0.891561, 1.024744]. The original failure is
retained. Both audits were replayed from a clean `e69828a` source archive and
these raw reports on Linux: outcomes and decision fields agree; nine controlled
standard deviations differ only at their last floating-point digit, so the
controlled audit JSON is not byte-identical across the Python environments.

The 45-process post-gate CPU follow-up passed its workload and source checks.
Median whole-process CPU time candidate/prior Rust was 0.592 for codec, 0.892
for BERT, 0.932 for GPT-2, 0.938 for Qwen3 and 0.914 for TinyStories. These
include startup, loading, all trials, reporting and teardown; they are not
per-request CPU or allocation counts. Python results and individual samples are
in the qualified summary in the archive.

Activity sampling found WindowServer in the top-CPU samples with median/max
13.6%/44.0% in the matrix (105 samples) and 13.5%/41.5% in the control (635
samples; absent from top entries counted as zero). No Chrome process appeared
in those sampled top entries. Neither before nor after thermal observations
reported thermal or performance warnings. These CPU snapshots cannot establish
exclusive GPU use or rule out unobserved activity. Resource and memory activity
samples and thermal observations are also retained.

## Matched memory diagnostic: unresolved

Three processes each for prior Rust, candidate Rust and qualified Python used
B1/B2/B4 and four scopes. Across all nine, fixture and output ID hashes match
the gate, latent float-byte hashes match across roles, and active MLX memory
returns to zero after model teardown. The recorded build/source and binary
hashes, complete commands, 48 snapshots per process, activity and thermal
observations are retained. The [offline checker](apple-gate18-memory-review-v1.py)
verifies report hashes and these identities.

Only the B4 resident decoder block immediate active snapshot exceeds 1.05:
69,702,948 candidate bytes versus 65,835,532 prior Rust and 65,835,528 Python
(1.058744 in each comparison). Immediately after the extra same-device barrier,
representative active bytes are 59,970,832 versus 59,970,056 and 59,970,052;
after 100 ms they are 59,970,828 versus 59,970,052 and 59,970,052. No delay
is excluded from the report. The post-barrier observations are consistent with
prompt cleanup, but identity of retained roots does not prove identical internal
live graphs or the ownership of each immediate allocation. The separate 1.05
comparable steady-state live-memory criterion remains **unresolved**, not
passed or failed by this diagnostic.

The B4 encode peak active MLX allocation was 155,866,252 candidate bytes versus
129,751,100 prior Rust and 129,750,040 Python (1.20127 and 1.20128). This is a
real observed peak-counter difference, distinct from steady-state active memory,
cache, host RSS and physical device-wide usage; it is not one of the 78 host
memory checks. Its practical effect requires investigation. A prior [matched linear-gather
experiment](mlx-linear-poc-v1.md) supplies a strong attribution: with all other
code held to the preceding native implementation, its three B4 encode peaks
were 129,613,076 bytes before the interpolation change and 155,857,940,
155,857,940 and 155,866,308 bytes after. That 26.2 MB increase closely matches
the new comparison. The same experiment reported a 0.552 B4 encode latency
ratio, so reverting it would sacrifice a measured performance benefit. This is
strong evidence for the interpolation tradeoff, not proof of the exact internal
buffer ownership. The performance control alone does not close Phase 2.

MLX's [0.32.2 Metal evaluation source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/eval.cpp)
retains input buffers in command completion handlers, and its
[allocator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp)
subtracts active bytes when buffers are freed. This offers a plausible mechanism
for the immediate-versus-barrier change; it does not establish attribution of
these measurements or justify modifying production synchronization. An
[upstream report](https://github.com/ml-explore/mlx/issues/4835) describes
unsynchronized allocator-counter reads; the 0.32.2
[getter](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.h)
likewise reads the counter without taking the writer mutex. This is another
limitation on interpreting concurrent immediate snapshots, not proof that the
observed values are wrong. No MLX pin or acceptance limit changed.

[Raw gate, control, resource, memory and activity evidence](raw/apple-gate18-resource-memory-v1.tar.xz),
SHA-256 `524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6`.
To reproduce the memory review, extract this archive into a clean checkout and
run `python3 docs/results/2026-11-11-phase2-optimization/apple-gate18-memory-review-v1.py .`.
