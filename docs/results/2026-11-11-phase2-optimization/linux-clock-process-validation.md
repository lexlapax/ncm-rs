# Clock process-record validation

**Status: metadata consistency component, not completed clock postflight. Linux
72/78; Apple pending.** `tools/bench/verify_clock_processes.py::validate_processes`
independently checks the eighteen-job sequence against separately frozen jobs,
the [compiled library contract](linux-clock-loader-preparation.md#compilation-of-the-observation-library-contract),
a declared forbidden injection identity and an independently supplied original
work deadline. It does not invoke the observation producer/factory to validate
its own records. Low-level numeric, clock-envelope and digest helpers are shared
with existing validators; source/producer authentication remains outside this API.

## Checked surface

- Exactly eighteen complete jobs in the fixed runtime/repetition order, with unique
  child PIDs and report paths. No partial prefix or copied report counts as another
  process. Receipt argv/cwd/report paths must match the supplied job declarations;
  taskset, workload image spelling and CPU2 prefix match the compiled contract.
- Complete owned-child receipts: zero exit, EOF, reaped child, no interruption or
  termination, declared 590-second timeout, zero termination grace, 1 MiB capture
  and 16 MiB report bounds. Capture byte accounting and report path/hash metadata
  must agree; **raw file contents are not read or independently hashed here**.
- Original deadline equality and finite owned-spawn evidence; sequence/receipt ends
  must not exceed that deadline. Receipt envelopes are nested and non-overlapping,
  with matching collector PID/parent declarations. Samples lie within their own
  receipt envelopes. UTC/monotonic domains use the existing 1,000 ns quantization;
  no clock tolerance is widened. This is not restoration/600-second lock validation.
- Complete observations with no latched failures or stronger provenance claims;
  bounded samples/transients, valid counters and at least one workload sample.
  Existing 3,000-sample, 50 ms interval, 8 MiB mapping and 16 MiB observation-record
  limits are retained.
- Recomputed mapping digests, exact mapping fields, sorted unique paths/non-file
  labels, kernel-link flags and kernel/path inode binding. Kernel-map device and
  pathname device remain separate; comparison uses the latter for prepared files.
- No sampled declared injection identity or path. Known library **and prepared
  image** paths must match the supplied identities from their first appearance;
  no changed known path is silently downgraded to an unlisted dependency. Mapping
  identities cannot change between samples/jobs. Every mapping set must be used.
- Sampled launcher/workload images, workload CPU2 and all runtime-specific mandatory
  library identities. Known optional libraries may remain absent. Unknown paths
  are retained separately, excluding prepared executable images from that category.

Only `process_receipt_consistency_verified` is true. Returned bindings include
canonical receipt/observation digests, all observed file metadata, known/prepared/
unlisted path classifications and the consistent clock-offset interval. Producer
identity, immutable mapped bytes, restoration, telemetry coverage, output quality
and frozen-file rechecks all remain explicitly false. No success here authorizes
file rehash before the remaining lifecycle/telemetry/quality gates have finished.

The caller must supply the actual reviewed contract/jobs/deadline, not values
reconstructed from collected reports to make them agree. This verifier cannot
prove that the expected declarations or receipts came from trusted producers.
Required-set semantics come from the frozen compiler; this component does not
recompile preparation records or independently authenticate their listings. Same-
inode byte changes, arbitrary aliases, unsampled/transient mappings and whole-life
injection absence still are not established. File contents and additional-library
roots/bounds belong to the separate final file postflight, which is unfinished.

## Verification evidence

Six new tests include **40 negative variants** across map tampering, record/order/
PID/command/report/scope failures, original deadlines/clocks/affinity/sampling and
contract/planned-path conflicts. Mapping-negative tests recompute the digests:
matching hashes cannot hide wrong facts. The overlap fixture keeps its altered
receipt/sample clocks internally consistent, so serial-envelope validation itself
must reject the overlap. Optional-absence and kernel/path-device distinction cases
remain positive. A no-file-IO/no-subprocess check covers metadata validation.

The integration test runs eighteen real fresh **CPU-only fake workloads** through
the existing bounded sequence and process observers. It reads the persisted
`sequence-plan.json`, `sequence-result.json`, all eighteen `.receipt.json` and
`.process.json` files, checks their equality with producer returns, and passes the
parsed records into the independent validator. All child PIDs are absent afterward.
These are temporary synthetic codec reports and minimal library declarations,
not the actual CUDA codecs/Python Torch workload or the compiled forty-library
hardware contract. No Torch import, CUDA validation, benchmark or policy action ran.
The temporary evidence is recreated by the test; it is not a real diagnostic run.

```sh
PYTHONPATH=tools/bench python -m unittest test_verify_clock_processes -v
PYTHONPATH=tools/bench python -m unittest test_verify_clock_processes test_clock_process_observation test_verify_readonly_processes -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **23 focused tests and default checks with 344 reporting tests pass**. Initial
and intermediate green runs are retained, including the final fixture enhancement
that reads the serialized per-job evidence rather than only producer return values.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-process-postflight-tests-v1.log` | `d1d84923badbf8264be87db3157efcdfd948a7a56c1ded5abb6d79cd869c4330` |
| `phase2-clock-process-postflight-tests-v2.log` | `18f087cb05f71a5276b6260769b4bcda03f67dd8f99455929c828fb7640f6169` |
| `phase2-clock-process-postflight-check-v1.log` | `4b364246bf79889cac8903a9080c8636ff1e57c8c07ca805add2c67a72b6c5fc` |
| `phase2-clock-process-postflight-tests-v3.log` (final) | `d4cb4864e6307423a5d24fb0a21210f61bb31d646243d63bd6c5fd80b95e689f` |
| `phase2-clock-process-postflight-check-v2.log` (final) | `2021c79c4ce83878316e105baed5fa6221f23b24a114313c8fff983f7de59a29` |

## Remaining gates

Current upstream verification, fresh complete preparation and frozen launch roles
remain pending. Bind restoration/observer lifecycles, full telemetry, report/output
quality and capture files, then recheck all frozen files and path bindings **last**.
This function checks sequence-local clocks, not the independent restoration
supervisor, set/reset commands or post-reset hardware state. No real collection,
permission test or policy change occurred. Default-policy/no-controller confirmation,
permissions/enforcement and a later fresh quiet window remain unresolved. Neither
this verifier nor its tests close a Linux acceptance cell or the Phase 2 objective.
