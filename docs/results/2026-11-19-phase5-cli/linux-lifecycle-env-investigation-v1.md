# CPU lifecycle fixture environment mismatch investigation

2026-11-20. A recent full `tools/check.sh` failed the CPU-only `test_real_writers_and_owned_process_samples_compose` with `capture_error:ObservationMismatch`; a focused retry and full retry passed, but that did not establish the cause. This investigation runs the existing fake lifecycle fixture only: no GPU workload, clock or machine policy changes.

An uninstrumented series completed 21 attempts and failed on the 22nd at job 4 with `live child environment mismatch`, zero accepted samples. A separate instrumented run using the committed probe completed five attempts and failed on the sixth at job 1 with the same reason and zero accepted samples. For that failing observation, the child's sampled environment equaled a subsequent read of the collector's `/proc/self/environ`; it lacked the two expected fixed fixture keys `LANG` and `LC_ALL`, while the fixture's `PATH` and `PYTHONNOUSERSITE` values matched. The observer and worker reported failure and owned-child cleanup completed. The probe records only fixed-key booleans, counts and environment-equality booleans, never environment values or unexpected key names. Its privacy and output-preflight tests passed.

The matching collector environment is consistent with a pre-exec/exec-transition observation, but does **not** prove the sampled state was pre-exec, explain every earlier failure, or establish a safe startup barrier. Prior independent collector-command and `PF_FORKNOEXEC` investigations and limitations are recorded in the [clock role plan](../2026-11-11-phase2-optimization/linux-clock-role-plan.md#collector-command-investigation). This work changes no production observer, acceptance, retry, signal, privilege or cleanup rules; mismatches still fail closed. The intermittent full-check failure remains unresolved.

The committed probe SHA-256 is `190549639dcfa0efefa6137627b53c7b82e7d94b1097fdbee905f93680299770`. The ignored instrumented summary SHA-256 is `0fd6fc0f2f44207374a88fefee23120067a876f18d05e94c8dd7f6ce99b80c90`; its failing process observation is `f1355d96094e5882dd7c2da34bc8dbaadf2287d3f2103f3b5180c935a9d0ff18`. Raw CPU-only fixture evidence remains under `reports/phase5-lifecycle-env-probe-v1/`; it is not durably published. A subsequent full `setpriv --inh-caps=-all tools/check.sh` passed, without resolving the intermittent mismatch.

## Test-only post-exec readiness

The CPU lifecycle-composition fixture now gives each trusted fake Python child its own marker written after that child starts executing, and delays its fixture-only observation callback until that marker exists. The real observer, collection and mismatch rules are unchanged; missing observations still fail through the real observer's `finish`, and an injected mismatch after readiness still propagates. This marker is **not authenticated and is not a production startup barrier**. The fixture no longer tests the pre-exec observation window; the separate real-observer mismatch/collector-command tests continue to require fail-closed behavior in that window.

Thirty consecutive executions of the real lifecycle-composition test passed with fixture markers (ignored summary SHA-256 `70821b8bf09dc74d4098c160cd1afe3d66fb904d9d699dafa54ee2841a9486d6`). The six lifecycle tests, 13 direct observer tests, nine coordinator tests and a full `setpriv --inh-caps=-all tools/check.sh` passed. The probe was updated to request the **unbarriered** fake fixture explicitly; 30 further attempts did not reproduce the intermittent mismatch (ignored summary SHA-256 `a48ab4074291093b34d834066cb90f37c374a5773521ba679f0e8ef7a6cc1011`). That is not evidence that the production observer race is resolved. Updated probe SHA-256 is `75c19ad805b08ee01194dd1c0b15f680ce2c5f3784bb79c4e403eb85f0e93a11`. The previously observed failures and their original probe identity remain on record above; a production-safe readiness design remains open.

Reproduce the unbarriered probe with a fresh output path (the failure is intermittent; 30 successful attempts would not disprove it):

```sh
setpriv --inh-caps=-all python3 tools/bench/phase5_lifecycle_env_probe.py \
  --output reports/NEW-lifecycle-env-probe --attempts 30
```

## Handoff executable-identity investigation

A later `tools/check.sh` failed a different CPU-only handoff fixture at job 8 with `unexpected live executable identity`; the process observer failed closed. The focused and full retries passed ([record](cuda-wmt14-pool-diagnostic-v1.md#collector-integration-smoke)). Its cause is not established; it must not be assumed to be the previous environment mismatch.

A test-only probe wraps the fake fixture worker's `ProcessObservation.observe` and, only when that exact mismatch is raised, records whether each already-sampled `/proc/<pid>/exe` inode matches the collector, expected workload or launcher. It records booleans only, never unexpected paths, argv, or inode values; it restores `os.stat` and re-raises the original exception. This wrapper may perturb timing, and matching an identity would not itself prove a trustworthy startup barrier. It does not change production observer acceptance, launch, privileges or machine policy.

Fifty instrumented full fake handoffs (18 jobs each, with skipped tests treated as failure) completed without reproducing the mismatch; no identity categories were observed. An earlier 50-run version also completed without a failure, and its logs were checked to confirm tests ran rather than skipped. The first one-attempt smoke had a test-probe quoting `SyntaxError` before the fixture ran; the corrected smoke passed. These negative results neither establish cause nor resolve the race. Final probe SHA-256 `73feeccd3aa27529e98e691669f2381615bec191b2e2db0b7f8d6641f2bb7332`; ignored 50-run summary SHA-256 `7186355d348714d0d4428e05a07f1e6befe92c4b1587907e66cb9e583c8d7df4`. Three probe tests, including mismatch preservation and output privacy, and `setpriv --inh-caps=-all tools/check.sh` passed. Raw fixture logs remain local under ignored `reports/phase5-handoff-image-probe-v2/`.

```sh
setpriv --inh-caps=-all python3 tools/bench/phase5_handoff_image_probe.py \
  --output reports/NEW-handoff-image-probe --attempts 50
```
