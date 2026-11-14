# TinyStories candidate: Apple quiet-window gate

The user selected a separately declared quiet-window gate for the unchanged
candidate; the [plan](tiny-borrow-apple-quiet-plan-v1.md) was committed before
collection. The initial matrix passed **77/78**: GPT-2 encode versus previous
Rust missed its process-median envelope. The unconditional full control also
passed **77/78**: that GPT-2 cell passed, while batch-4 resident decoder block
p95 versus previous Rust remained open. Its control point ratio was **1.05673**
and the paired-process percentile interval's upper endpoint was **1.10442**
against the unchanged **1.10** limit. The equivalent upper endpoint versus
Python was **1.09610**. The previous control remains **76/78** and is not
superseded or erased by this observation.

The initial and control collections completed all 45 and 135 processes,
respectively. The initial activity log has 106 samples, WindowServer
median/peak 1.45%/2.8%, no Chrome, and one wallpaper appearance; the control
has 319 samples, 2.05%/4.6%, no Chrome or wallpaper among sampled top processes.
No thermal warning was recorded before or after either collection. These
samples neither prove exclusive GPU access nor identify a cause. One launch
before collection failed while recording metadata because noninteractive SSH
could not find `rustc`; no workload ran. Its empty output directory was preserved
on the Mac, and the process-local PATH was corrected for the collection.

The benchmark binaries, native libraries and runner match the previous gate;
all 300 current report source inventories match their corresponding previous
gate inventories. There are no changes to runtime, tools, pins or fixtures
between `3555c67` and the measured documentation-only commit `eb051c7`.
Existing trained qualification and lifecycle evidence are reused on that basis.
The controlled audit reproduces byte-for-byte from a clean export: SHA-256
`fbc356cc28590b64dd11c07705e24f360601b69e902e5f62731757d766c558eb`.

## Rethinking the remaining work

The repeated failures warrant more than another unchanged gate. In the quiet
control, candidate process 1 had a batch-4 block median of 1.3038 ms and maximum
of 2.6389 ms; the other eight process medians ranged 1.1178–1.1397 ms. In the
previous control the conspicuously slow candidate was process 8 instead. This
is an exploratory observation, not grounds for dropping either process. Across
all 270 candidate trials, pooled median/p95 were 1.13792/1.32650 ms in the quiet
window versus 1.13142/1.34054 ms previously. Prior Rust's quiet-window median/
p95 were 1.21929/1.25529 ms. Although the candidate's median improves, these
observations do not establish a reliable p95 improvement or the cause of the
tail. The earlier instrumented graph/evaluation split also cannot separate GPU
execution from submission and waiting.

The current 10,000-resample percentile interval resamples nine whole sequential
process triplets, preserving process-level variation but providing a descriptive
interval rather than a simultaneous guarantee. Counterbalancing does not make
the machine conditions of sequential processes identical. Do not infer a
defective algorithm, GPU contention or a bad gate solely from these data, and
do not widen the limits or choose a favorable window. Next investigate the
recurring slow-process behavior with focused host-scheduling and Metal
submission/execution attribution, then evaluate an evidence-backed runtime
change or a separately reviewed measurement design. A new full acceptance gate
should follow a substantive change or an explicitly justified new protocol, not
be used as a search for a lucky pass.

[Raw reports, activity, qualification, lifecycle, configurations and audits](raw/tiny-borrow-apple-quiet-gate-v1.tar.xz)
have SHA-256 `411f5555a7949003be115239dc03f5b3e4060c5c75350b52b5cf46a19c30ac88`.
Reproduce from a clean checkout by extracting the archive at its root and running
`python3 tools/bench/audit_phase2_performance.py reports/quiet-controlled-config.json <new-output>`.
