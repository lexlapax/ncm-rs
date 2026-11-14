# Decoder trace-label elision — rejected

An isolated candidate avoided formatting linear and normalization trace labels
in untraced decoder calls. Traced calls retained their labels; model arithmetic
and public APIs were unchanged. The candidate passed `cargo test -p ncm-model
--locked` and was built as a release CUDA `codec_bench`. The exact rejected
[patch](raw/decoder-trace-label-v1.patch) has SHA-256
`6885d90e71f0233bd808f906224397a13377e342f4e0ebb79592fab6f45b2e0a`.

The saved geometry candidate and modified binary ran three fresh processes each,
in order previous/candidate, candidate/previous, previous/candidate, using CPU 2,
the existing codec workload, five warmups and thirty trials per scope. The GPU
reported zero utilization before the run; this was not a certified quiet window.
Median of the three per-process B1 resident-block medians was 0.17933 / 0.17913 ms
(previous/candidate); B1 reconstruction was 1.97773 / 1.97609 ms. B2 and B4
results likewise showed no material benefit; output-ID hashes matched for all
twelve batch/scope combinations. The [raw reports and binary hashes](raw/decoder-trace-label-v1.tar.xz)
have SHA-256 `f53d76b34dda32a08ac5b18cdc60f920ad128da1e51731a92729ddb156990071`.
No full numerical trace, Python comparator, lifecycle, acceptance audit or Apple
test was run for this rejected candidate. It was reverted; the current release
benchmark executable still reflects it and requires rebuilding before reuse.

This limited diagnostic supports rejecting this implementation, not a general
conclusion about every trace or weight-name optimization. It does not change the
Linux 72/78 acceptance result or the requirement for both platform gates.
