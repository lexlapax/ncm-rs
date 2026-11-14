# Focused Metal tail-trace feasibility

After the [quiet-window gate](tiny-borrow-apple-quiet-gate-v1.md), one bounded
diagnostic process ran under the existing Xcode **Metal System Trace** template,
with `--no-prompt` and a 45-second limit. It exited successfully in 12.34 seconds.
The executable was the previous instrumented block-segments diagnostic, SHA-256
`3fb22a460b7c31af1c1626b2ccfdf0cbed015d49e71f7571c25bc96ac4237acc`;
its source, workload, manifest and scope input/output hashes match that
[earlier diagnostic](mlx-block-segments-v1.md). Tracing changes timing and is
not an acceptance measurement.

The trace exposed a GPU-interval table with 9,171 records, including interval
duration and CPU-to-GPU latency fields. No attribution of those records to a
specific block trial is claimed. The batch-4 block's 30 measured trials had
median **0.957625 ms**, maximum **0.997959 ms**, and none exceeded 1.2 times
the process median. Consequently this run did not capture the target slow
process and does not explain the earlier tail or establish a performance fix.

The existing diagnostic reports durations but not trace-correlatable trial
timestamps or markers. A useful next diagnostic must add explicit trial
boundaries and verify alignment with the trace, then capture an actual slow
trial before comparing host, submission and GPU intervals. Do not treat GPU
intervals as the whole wall time or repeatedly rerun the acceptance gate.

The [diagnostic report](raw/mlx-tail-metal-feasibility-v1.tar.xz) has SHA-256
`d679cfe03c0916f4c44a974e6f6cbbc1ca04288c6fa8a5316c1889cc58b78b95`.
The full system trace remains on the Mac at
`.reference/phase2-mlx-tail-metal-v1.trace`; it includes system process metadata
and is not included in the published archive. The exported GPU table is a
temporary local file on the Mac. No clock or power changes were made.
