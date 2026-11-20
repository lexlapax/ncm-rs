# CPU lifecycle fixture environment mismatch investigation

2026-11-20. A recent full `tools/check.sh` failed the CPU-only `test_real_writers_and_owned_process_samples_compose` with `capture_error:ObservationMismatch`; a focused retry and full retry passed, but that did not establish the cause. This investigation runs the existing fake lifecycle fixture only: no GPU workload, clock or machine policy changes.

An uninstrumented series completed 21 attempts and failed on the 22nd at job 4 with `live child environment mismatch`, zero accepted samples. A separate instrumented run using the committed probe completed five attempts and failed on the sixth at job 1 with the same reason and zero accepted samples. For that failing observation, the child's sampled environment equaled a subsequent read of the collector's `/proc/self/environ`; it lacked the two expected fixed fixture keys `LANG` and `LC_ALL`, while the fixture's `PATH` and `PYTHONNOUSERSITE` values matched. The observer and worker reported failure and owned-child cleanup completed. The probe records only fixed-key booleans, counts and environment-equality booleans, never environment values or unexpected key names. Its privacy and output-preflight tests passed.

The matching collector environment is consistent with a pre-exec/exec-transition observation, but does **not** prove the sampled state was pre-exec, explain every earlier failure, or establish a safe startup barrier. Prior independent collector-command and `PF_FORKNOEXEC` investigations and limitations are recorded in the [clock role plan](../2026-11-11-phase2-optimization/linux-clock-role-plan.md#collector-command-investigation). This work changes no production observer, acceptance, retry, signal, privilege or cleanup rules; mismatches still fail closed. The intermittent full-check failure remains unresolved.

The committed probe SHA-256 is `190549639dcfa0efefa6137627b53c7b82e7d94b1097fdbee905f93680299770`. The ignored instrumented summary SHA-256 is `0fd6fc0f2f44207374a88fefee23120067a876f18d05e94c8dd7f6ce99b80c90`; its failing process observation is `f1355d96094e5882dd7c2da34bc8dbaadf2287d3f2103f3b5180c935a9d0ff18`. Raw CPU-only fixture evidence remains under `reports/phase5-lifecycle-env-probe-v1/`; it is not durably published. A subsequent full `setpriv --inh-caps=-all tools/check.sh` passed, without resolving the intermittent mismatch.

Reproduce with a fresh output path (the failure is intermittent; 30 successful attempts would not disprove it):

```sh
setpriv --inh-caps=-all python3 tools/bench/phase5_lifecycle_env_probe.py \
  --output reports/NEW-lifecycle-env-probe --attempts 30
```
