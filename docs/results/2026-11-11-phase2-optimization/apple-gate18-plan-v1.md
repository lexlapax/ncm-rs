# Apple independent 18-process control

After reviewing the repeated failures and confirming the Mac is ready, the user
directed the longer Apple gate. Run the unchanged qualified runtime `5200924`
against accepted Rust and qualified Python. First collect a fresh full matrix
with three processes per runtime/workload, then unconditionally collect one full
18-process control per runtime/workload if collection succeeds. Five workloads,
three runtimes and 21 repeats produce 315 benchmark processes. The existing
runner uses five warmups and 30 timings per process, rotates runtime order,
records CPU activity and thermal state, and retains every report. This takes
roughly 25 minutes based on earlier runs. No outlier removal or extra retry is
planned. Earlier 76/78 and 77/78 controls remain evidence.

The [controlled config](apple-gate18-controlled-config-v1.json) feeds all five
control workloads into the unchanged 78-cell audit, with existing median, p95,
memory, quality and output limits. The [initial config](apple-gate18-initial-config-v1.json)
uses the fresh three-process matrix. Qualified trained trace and lifecycle
evidence may be reused only after identity verification. Before collection the
wrapper checks the clean current and baseline checkouts, baseline commit,
production-source identity, executable/library/runner hashes, Python packages,
OS/compiler versions, artifact manifest, and restored qualification/lifecycle
hashes. It fails before collection if an identity differs or an output exists.

The predeclared [runner](apple-gate18-run-v1.py) executes the matrix and control,
then both audits. A failed collection is preserved and stops the sequence; a
failed audit remains a failed result. The 18-process audit path was checked with
an [offline synthetic replay](mlx-all-windows-exploratory-v1.md#existing-tooling-offline-18-process-replay),
which was not a gate and does not predict this outcome. Activity observations do
not establish exclusive GPU access. No code, workload, numerical policy,
clock/power setting, or global configuration changes are part of this run.
