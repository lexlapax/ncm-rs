# Approved decoder candidate Linux gate — 70/78

Candidate `211df63` includes the approved byte-level decoder change. Its exact
source passed the full CUDA and MLX checks in isolated review worktrees; the
main-workspace tokenizer suite and all four Linux asset probes pass. CUDA codec
benchmark, trace and lifecycle executables are byte-identical to the previously
qualified versions, so their 642-trace / 2,500-decision quality result is reused.

The fresh 45-process Linux matrix passes **71/78** initially. Qwen decode and
skip-special decode pass versus Rust with median ratios **0.7376/0.7358** and
p95 ratios **0.7318/0.7446**. Seven other cells fail point estimates or the
process envelope. One fixed, declared-before-collection follow-up ran nine
fresh processes per runtime across every codec, BERT, GPT-2 and TinyStories
scope and checked load. The controlled audit passes **70/78**, including
the Qwen cells, but reveals additional interval failures in the controlled
scopes. No initial or control measurements were discarded and no limit changed.

The eight controlled failures are batch-1 codec decode versus Rust; batch-2
and batch-4 resident block versus both Rust and Python; BERT skip-special
decode versus Rust; TinyStories decode versus Rust; and BERT checked load
versus Rust. Codec binaries are identical to the prior passing candidate,
but that does not establish the cause of the observed variation or make this
candidate's failed gate pass. GPU telemetry contains 170 matrix and 364
control samples; SM clocks ranged from 210 to 1,965 MHz and reported GPU
utilization ranged from 2% to 99% during both captures. These sampled values
include benchmark work and do not establish per-request causality or exclusive
GPU access. No clock or power policy changed.

[Raw matrix, control, plans, telemetry, probes, lifecycle and audits](raw/decode-final-linux-gate-v1.tar.xz)
have SHA-256 `d5be6edf217f9d613aacf001c1d3be5e955ff63780a8c7b6143598f26d3cc773`.
The controlled audit SHA-256
`135b1ad81fbaccca982f278dcd928cc1455a4b7719a0d9772be0e0efbc27973d`
reproduces byte-for-byte, including the eight failures, from a clean `211df63`
export and the archive. The next action is to diagnose the observed timing
instability; the Linux-first rule prevents advancing to the Apple gate.
