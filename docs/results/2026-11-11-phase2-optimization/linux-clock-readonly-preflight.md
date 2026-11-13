# Read-only clock preparation preflight

`clock_preflight.run` accepts two explicitly hash-bound private reviewed records:
a file-only snapshot and read-only GPU metadata. It requires a fresh single-threaded
Linux `-S` interpreter and its own reviewed checkout/Python image. It generates a
new file snapshot and GPU metadata, compares the stable fields and utility identity,
then repeats the worktree guard and rehashes all declared files/aliases **after**
the GPU queries. The new GPU record is exclusively written and re-read by hash;
its digest and the current snapshot digest are included in the private preflight
result. Output and collection paths must be fresh and disjoint.

This is an import-only **read-only preflight**, not a launch entry. It performs
bounded Git/loader/NVIDIA identity and supported-clock probes but never runs a
benchmark, starts telemetry, changes clocks or checks privilege by mutation. It
leaves producer authentication, role ancestry, permissions, hardware policy,
restoration, availability and readiness false. The caller-supplied reviewed hashes
and the future live entry must still be authenticated. File/path checks are sampled,
not atomic or proof of mapped bytes. A successful comparison would not close any
Linux acceptance cell. The user has not supplied a fresh diagnostic quiet window.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_preflight test_compare_clock_snapshots test_compare_clock_gpu -v
```

Ten updated focused tests pass, including bad reviewed digest, each producer/
comparison/late-check failure, reviewed-evidence output isolation and worktree
drift; the tests stub producers and do not query the GPU. An earlier 19-test
focused run also covered snapshot and file-check regressions. Private current-GPU
record publication and reviewed-evidence path isolation were added afterward and
passed the updated rerun. No actual assembled preflight has yet been run;
a clean committed snapshot from the final relevant source state is necessary.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed the default checks,
including 436 reporting tests, before the two refinements described above; their
updated focused tests passed afterward. Local log:
`/tmp/ncm-phase2-clock-preflight-check.log`.
