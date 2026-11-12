# Clock telemetry-content validation

**Status: offline component, not actual clock collection. Linux 72/78; Apple pending.**
`tools/bench/validate_clock_telemetry.py::validate` checks supplied full CPU/GPU bytes,
readiness, telemetry receipts, the restorer outer receipt and eighteen job windows.
It reuses `validate_readonly_telemetry.cpu_records` and `gpu_records` **unchanged**;
the old read-only twelve-job validator and its output remain unchanged.

## Checked surface and limits

- Exactly metadata + **901 CPU samples + completion**, reviewed sampler source,
  CPU2 workload/CPU0 observer/CPU3 SMT-sibling topology, tick fields/arithmetic,
  sample schedule/lateness, UTC/monotonic intervals and at least 900-second completion.
  Optional-data absence and unavailable counter deltas are retained. Shared input
  bounds stay 16 MiB/raw stream; no numerical tolerance or sampling-gap gate changes.
- GPU UUID, CSV header/columns/units/numeric fields, increasing timestamps and
  unambiguous local timestamps under the separately declared IANA timezone. GPU
  sampling gaps and unavailable values remain explicit; no sample count is invented.
- CPU raw size/hash/path versus the post-exit report receipt, expected CPU argv,
  complete telemetry status, and GPU raw SHA-256 versus its telemetry record.
  These bind supplied bytes to supplied receipts, **not** to authenticated producers
  or immutable filesystem paths. Raw-file identity/capture checks remain separate.
- Readiness worker/GPU IDs and CPU0 affinity; CPU metadata must exactly match the
  raw stream, and the first GPU row must match the actual first parsed CSV row.
  Readiness's explicit local offset must match the declared zone. Its local timestamp
  lies after both first sample timestamps and before its recorded UTC clock stamp;
  the recorded bracket is within the CPU receipt and before restorer capture.
- CPU samples lie inside their capture lifetime. All 901 sample clocks, readiness,
  outer receipt and job receipts share the unchanged UTC/monotonic quantization.
- One complete restorer window plus exactly eighteen distinct complete job receipts,
  fixed runtime/repetition order, no overlapping jobs, and every job inside the outer
  restorer window. Both CPU and GPU sample endpoints bracket **the entire restorer
  capture**, not just the jobs. CPU/GPU interval indices and inside-window GPU sample
  indices are retained for every window.

Returned canonical digests bind sequence, restorer receipt, telemetry, readiness and
parsed CPU samples for composition with the [sibling lifecycle validator](linux-clock-restoration-validation.md#sibling-lifecycle-follow-up).
This validator reports content/readiness/receipt-window consistency only. It does
not establish live ownership, safe ancestry, role images/environments, clock-lock
permission/enforcement, hardware restoration, continuous clock/quietness coverage,
codec output quality, performance acceptance or finalization ordering. In particular,
sampled SM frequencies are retained data, **not** an enforcement verdict. ZoneInfo
may read timezone data; raw-byte hashing here is not frozen-file rehash.

## Tests and retained regression

Six new tests cover **43 negative variants**: raw-receipt/ready/declaration mismatch,
rehashed malformed/short/truncated CPU data, rehashed GPU unit/UUID/time failures,
restorer-only coverage deficits, and incomplete/reordered/overlapping job windows.
Positive virtual-clock tests run the actual CPU serializer over fake procfs for all
901 samples; no 900-second real collection occurs. Sparse GPU input with 500/400-second
gaps and unavailable SM/power fields remains visible and explicitly cannot establish
clock enforcement. The test does not invent an allowed physical gap threshold.

Review added two negative readiness cases after the initial green run. Both initially
failed with **`AssertionError: ValueError not raised`**: a readiness-local timestamp
before sampling began, and a first GPU timestamp after readiness while all job/outer
windows still fit. The new explicit first-samples ≤ local-readiness ≤ record-clock
check rejects both. The red log is retained; no shared parser or tolerance changed.

```sh
PYTHONPATH=tools/bench python -m unittest test_validate_clock_telemetry test_validate_readonly_telemetry -v
PYTHONPATH=tools/bench python -m unittest test_validate_clock_telemetry.ClockTelemetryTests.test_raw_receipts_readiness_and_external_declaration_mismatches -v
PYTHONPATH=tools/bench python -m unittest test_validate_clock_telemetry test_validate_readonly_telemetry test_validate_clock_lifecycle -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **21 focused and 370 reporting tests pass**. The focused set includes the
existing real short CPU/fake-policy sibling-lifecycle transport, not actual GPU
telemetry or codec qualification. No Torch import, clock command, benchmark, package
change or fresh installed-library/preparation scan ran. The earlier unrelated
lifecycle-fixture failure remains documented with cause unestablished.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-telemetry-validation-tests-v1.log` | `a2caeab0068639a176c676a9aa1bc6bbac2e1b3c1acf5e72382976091121cd82` |
| `phase2-clock-telemetry-ready-time-red-v1.log` | `2de2ae261d450699756fd874e5f25aa20d6960f075e4f016e3c29cec8ef9155c` |
| `phase2-clock-telemetry-validation-tests-v2.log` (21 pass) | `4eae81db3090e4e3f387a12faf28635eea1091d07c23636bfc7dacfa992c9bb2` |
| `phase2-clock-telemetry-validation-check-v1.log` (370 pass) | `a235f0f83a8a9d7aa965488a93f7babe68479a44559ee0a00fe0e3f63791ee9c` |

## Next boundary

Report/output and raw-capture qualification, then full finalization binding every
gate to the same records/files and enforcing frozen rehash last. Current upstream
verification, fresh complete preparation, frozen role/entry binding and actual
policy/enforcement/restoration evidence remain unfinished. Default-policy/no-custom-
lock/no-competing-controller confirmation, permissions and a later new quiet window
are still missing. No live launch gate or acceptance comparison closes here.
