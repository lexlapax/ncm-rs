# XSum borrowed upload: repeated Metal comparison plan

After the full Linux CUDA candidate checks, output comparison and repeated
performance matrix, the ordered `f03650b` Metal check passed. The explicitly
rebuilt Metal CLI matched the 32-passage pinned Python CUDA reference. Two
32-row batches containing the known divergent rows reproduced the previous
Metal outputs exactly; the full raw-ID gate remains failed. The first short
memory screen began before the corrected Linux CLI check finished and is retained
as exploratory, not as an ordered gate.

The repeated performance comparison will use the same 32 verified input passages
at batches 1, 4 and 32. For each batch, run three fresh processes per runtime,
with alternating order, five warmups and 30 synchronized generations and code/ID
host readbacks per process. The Python MLX adapter is a qualified adaptation,
not the original PyTorch CUDA runtime. All codes and IDs must match between
runtimes and repetitions. Report process-median latency, pooled p95, process
peak RSS including load, and MLX active/peak/cache memory with their different
scopes. The adapter's explicit garbage collection happens after each timed
request, outside its recorded latency. Rust's median and pooled p95 must be lower,
as must its median process peak RSS, in every batch for this comparison to favor
it. Report system-wide GPU registry and CPU activity, source
and native-library hashes, exact commands and any limitations; registry polling
can perturb timing and does not establish exclusivity. Do not infer a full quality
or exact raw-ID pass from these 32 passages.

The harness adds explicit warmup/trial arguments while retaining the earlier
2/5 screen defaults. Validate Linux first, then build and check the harness on
Metal. Keep all failed runs and do not change a threshold or source identity to
hide a failure.

Linux validation passed the focused harness/contiguous-batch tests and full
`setpriv --inh-caps=-all env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 CUDA_COMPUTE_CAP=86
./tools/check.sh --cuda`, including 18 real-CUDA operator cases. The log
`reports/phase4-xsum-mlx-bench-preparation-cuda-check-v2.log` hashes to
`fbd217c3dc04b833384aa55103d279e4884f829c7331e43639d8fe0bd825fe5e`.
The initial sandboxed v1 run passed its reporting tests but failed CUDA
capability detection because `nvidia-smi` was inaccessible; its log is retained.
Metal compilation and exercise of both default and extended trial counts remain
pending.
