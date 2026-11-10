# Phase 2 Apple remediation — qualified gains, acceptance still open

Candidate **`083c1e5`**, M1 Ultra / native MLX 0.32.2. Linux validation preceded
every shared change. **Do not close Phase 2 or advance to Phase 3 yet.**

All **642 trained traces / 2,500 decisions pass**, zero ID disagreements, same two
near ties, unchanged CUDA oracle and tolerances. Native uninstrumented/traced
execution agrees. Full `tools/check.sh --mlx` and real tokenizer goldens/rejections
completed successfully on the Mac. Their latest detailed logs/results remain
remote pending restored SSH access; preceding full Metal logs are published in
the compact-tokenizer report. Trained qualification and capture metadata are
published here; large trace tensors remain ignored, not staged as model artifacts.

## Matched three-way results

Three fresh processes/runtime, five warmups, thirty measured trials, rotating
previous/Python/candidate order. Previous source is clean `5dc9626`, with preserved
historical binaries. Native and Python use the same pinned workloads/output
retention and explicit qualified adapted-Python MLX baseline. Python evidence is
shared between comparisons, not counted twice. Raw reports, binary/source hashes,
commands, power/thermal observations and CPU-activity telemetry are archived.

Pooled warm checked loads and maximum observed host HWM:

| Scope | Old / candidate / Python load (ms) | Old / candidate / Python HWM (MB, decimal) |
| --- | --- | --- |
| Codec | 945.868 / 179.987 / 272.205 | 4749.16 / 293.91 / 308.69 |
| BERT | 11.835 / 3.474 / 17.481 | 30.83 / 15.97 / 104.01 |
| GPT-2 | 51.254 / 17.027 / 50.907 | 64.47 / 24.99 / 228.61 |
| Qwen | 189.288 / 63.651 / 144.780 | 633.88 / 69.81 / 290.98 |
| TinyStories | 5.625 / 3.799 / 30.708 | 11.81 / 10.62 / 84.80 |

All formerly measured Apple load/memory deficits are removed in this matrix.
Codec load is approximately **5.26× faster**, HWM **16.2× lower** than old Rust;
Qwen load **2.97× faster**, HWM **9.1× lower**. These are implementation/API results,
not format-controlled parser or language-level claims. Prior failed intermediate
memory/timing results remain in the shared-loading and compact-tokenizer reports.

Codec pooled medians, old / candidate / Python milliseconds:

| Batch | Encode | Decode | Reconstruct | Resident block |
| --- | --- | --- | --- | --- |
| 1 | 13.1085 / 13.1570 / 15.2135 | 11.9641 / 11.9467 / 13.7672 | 24.8175 / 24.9826 / 28.7062 | 1.1126 / 1.1006 / 1.1164 |
| 2 | 14.1131 / 13.9917 / 16.0876 | 13.0292 / 13.0796 / 14.8220 | 27.0448 / 27.0685 / 30.8275 | 1.1787 / 1.1852 / 1.2002 |
| 4 | 14.9501 / 15.0685 / 17.0295 | 14.1771 / 14.2237 / 16.0701 | 29.0790 / 29.2383 / 32.9015 | 1.2159 / 1.2151 / 1.2326 |

Both complete summary JSONs contain p95, dispersion, first-call measurements,
throughputs, instrumented stages, comparable RSS/HWM and separate MLX counters.
No timing scopes were shortened; imports/context startup are not included here.

## Stability and uncertainty

Separate 36-load/drop diagnostics show bounded residency, with zero MLX active
bytes after every drop:

| Diagnostic | First / final dropped RSS (bytes) | Retained MLX cache (bytes) |
| --- | --- | --- |
| Authenticated host artifact only | 181,911,552 / 182,239,232 | 8 |
| Copied arrays for every declared tensor | 344,801,280 / 345,112,576 | 162,321,432 |
| Inference codec | 266,141,696 / 266,502,144 | 59,593,224 |

Array diagnostics intentionally include non-inference tensors; they are not the
matched inference benchmark. Allocator active/cache/peak and process RSS are
separate unified-memory observations, **not separate or interchangeable VRAM**.
The full benchmark also repeatedly executes all B1/B2/B4 scopes and validates
stable outputs. This supports bounded observed workloads, not a universal leak
absence guarantee.

The diagnostic comparison has **75/78 cells inside the full bands/envelopes**.
Every pooled median/p95 ratio and every comparable host-memory ratio passes.
Three execution uncertainty envelopes versus previous Rust remain above 1.05:

- GPT-2 encode: median 0.99888, p95 1.05757, envelope 1.06977.
- Qwen encode: median 0.99473, p95 1.03065, envelope 1.09389.
- Qwen decode: median 1.00786, p95 0.98641, envelope 1.08570.

These require additional fixed fresh-process controls, not widened thresholds or
selection of favorable runs. macOS has no CPU-affinity claim; live desktop activity
is disclosed and sampled. No unrelated process, clock, power or allocator policy
was changed. No privileged GPU utilization/temperature measurement is invented.

## Lifecycle and access blocker

A separate lifecycle baseline worktree was created at
`.reference/phase2-mlx-lifecycle-baseline` (`5dc9626`), adding only copies of the
reviewed lifecycle example and shared harness. Historical baseline/candidate
worktrees were untouched. Old native lifecycle build completed, and the remote
three-way lifecycle command returned successfully. **Its report has not yet been
retrieved or independently inspected locally.** The local SSH agent became
unavailable before transfer; a batch-mode retry was denied. No lifecycle figures
or final acceptance are inferred from the missing local report.

After access is restored, keep a process-local persistent SSH connection for the
remaining transfers/measurements (no global SSH/agent changes). Retrieve lifecycle
and final check logs, collect fixed GPT-2/Qwen controls, run the complete fail-closed
audit and committed-only reproduction, then publish the final platform decision.
The supplied config deliberately points at the still-missing lifecycle report;
the audit currently rejects it. `phase2-mlx-matrix-diagnostic.json` is explicitly
**not an acceptance record**.

## Reproduction and audit hardening

`evidence.json` fingerprints raw evidence and summaries. Unpack
`raw/threeway-v1.tar.gz` into `reports/phase2-performance-mlx-remediated-v1/`, then:

```sh
python3 tools/bench/phase2_mlx_summary.py \
  reports/phase2-performance-mlx-remediated-v1/candidate reports/apple-candidate.json
python3 tools/bench/phase2_mlx_summary.py \
  reports/phase2-performance-mlx-remediated-v1/previous reports/apple-previous.json
```

The acceptance audit now binds controlled **Python reference source hashes**, as
well as native ones, to the original matrix. A mutation regression rejects changed
reference code. The latest Linux acceptance still passes all 78 comparisons under
this stronger check (`raw/linux-bound-reference-audit.json`). This is reporting
hardening, not a production change or a relaxation of any gate. A clean staged
checkout without ignored reports, weights or reference environments exactly
reproduces this Linux audit and both Apple summaries; all 24 report-tool tests pass.
