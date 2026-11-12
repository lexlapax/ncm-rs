# Clock sequence evidence-file binding

**Status: post-collection reader component, not complete finalization. Linux 72/78;
Apple pending.** `tools/bench/clock_evidence.py` adds `Evidence` and `read_sequence`
for current private on-disk clock sequence evidence. No runner, benchmark, policy
command or frozen preparation-file rehash is invoked.

## Reader boundary

`Evidence(root)` requires an existing canonical owned private directory. Each
relative path is normalized and confined by descriptor-relative traversal: no `..`,
absolute path, empty path, backslash or symlink traversal. Root identity is rechecked;
intermediate directories and regular leaves must remain owned/private (no group/other
permissions). FIFOs and nonregular leaves cannot block an unbounded ordinary read:
leaves use `O_NOFOLLOW | O_NONBLOCK` and are type-checked before reading.

Each path is read once. Before byte reads, size/count admission is checked: **128
files and 1 GiB aggregate bytes per reader**, at most 16 MiB per general record/report,
and 1 MiB for job captures and receipt records. These are postflight read-admission
bounds, **not** collection quotas, filesystem quotas, JSON-memory/RSS limits or
hashing-time guarantees. No read-only v1 or workload capture budget is changed.

The reader hashes the exact bytes returned and retains pathname device/inode/size/
SHA-256. Descriptor metadata before/after and a reopened no-symlink pathname must
agree on device/inode/size/mtime/ctime/mode. `verify_unchanged()` checks every previously
read path again, detecting visible earlier-file changes during later reads. The
sequence reader invokes that final check; a composed finalizer can invoke it again
before the all-frozen-file rehash. No atomic snapshot or adversarial-swap immunity
is claimed. Trusted parent/producer setup and actual writer completion remain owed.
JSON parsing rejects duplicate keys/nonfinite constants. Failure returns no completed
binding and leaves source files untouched; this component writes no evidence files.

## Fixed sequence binding

`read_sequence(reader, jobs=..., work_deadline=..., sequence_dir=...)` reads exactly
**74 expected files**: plan/result plus eighteen report/capture/receipt/process-record
sets. It checks:

- Separately supplied eighteen-job plan, fixed runtime/repetition/report paths and
  original work deadline against persisted plan/result; unchanged capture/report/
  free-space declarations and diagnostic/deferred-copy scope flags.
- Complete fixed sequence, unique supplied process PIDs, expected argv/cwd and enabled
  observers. Each individual receipt and observation matches the embedded sequence
  record canonically, not just a producer's success summary.
- Capture raw length equals both observed/captured counts and its SHA matches the
  receipt, within 1 MiB. **Captures have no producer inode receipt.** A same-byte
  replacement before the first read is allowed by hash/length consistency and is
  explicitly not promoted to producer-inode or provenance proof.
- Reports retain the receipt's path, device/inode, size and SHA under the 16 MiB bound.
  Same-byte new-inode replacement therefore fails for reports. Post-exit receipt
  verification flags remain distinct from producer authentication.
- All read files retain their observed metadata through the final inventory check.
  Additional directory contents are not audited and do not count as more processes.

The returned parsed plan/sequence and exact report bytes feed existing validators;
the reader retains all 74 identities. Its binding claims only
`sequence_file_receipt_consistency_verified`, not process-map correctness, report
quality, complete role/lifecycle qualification, safe ancestry or enforced ordering.
In particular, matching an observation file to its embedded record does not prove
that the observation itself is valid. The [process validator](linux-clock-process-validation.md),
[lifecycle validators](linux-clock-restoration-validation.md), [telemetry checks](linux-clock-telemetry-validation.md)
and [output qualifier](linux-clock-output-qualification.md) remain independent gates.

## Verification

Eight new tests cover the 74-file layout, plan/deadline/order/command/PID drift,
individual-versus-embedded evidence mismatch, changed capture/report bytes,
same-byte report replacement, duplicate/nonfinite JSON, duplicate reads, invalid
paths/symlink parents/leaves, FIFOs/public permissions, root replacement, byte/count
admission before reads and mutation during/after earlier reads. A positive test
explicitly permits same-byte capture replacement before reading and keeps the
producer-inode claim false.

The actual CPU integration runs eighteen fake workloads with real process observers,
then reads their persisted files and compares the parsed sequence to the producer
return; all child PIDs are absent. Its temporary private umask is restored afterward.
A separate **fresh subprocess** preserves RSS-sensitive parent tests while feeding
all eighteen hash-verified archived v2 reports through newly written temporary
files/synthetic receipts, this reader and the actual unchanged-tolerance output
qualifier. That is historical-output transport, not a fresh codec run or salvage of
failed read-only v1. It does not qualify live GPU provenance or clocks.

```sh
PYTHONPATH=tools/bench python -m unittest test_clock_evidence test_qualify_clock_codec -v
PYTHONPATH=tools/bench python -m unittest test_clock_evidence test_qualify_clock_codec test_verify_clock_processes -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **17 focused and 381 reporting tests pass**. The initial green log predates
adding the positive capture-identity limitation check. No Torch/GPU/benchmark/policy
work or installed-library preparation scan occurred. Earlier failure evidence is
retained; unrelated lifecycle-fixture failure is not declared repaired.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-evidence-tests-v1.log` | `347ba51f307c885db4ca7958ae977126e7f0bb8394a42805be9455595e57604a` |
| `phase2-clock-evidence-tests-v2.log` (17 pass) | `0b60f40ec26e351de0733d1fd35d25856950b659375047d46a296c471e7aa0a8` |
| `phase2-clock-evidence-check-v1.log` (381 pass) | `c1b56ec8aae9529f26f5dfadc0640d0b5e4a74d9ed8d88c7b06924d47f938e1b` |

## Remaining boundary

Observer/restorer/coordinator role-command and outer-capture file binding, then a
same-evidence finalizer enforcing all validators before frozen rehash **last**.
Current upstream, fresh full preparation/frozen entry, default-policy/no-controller
confirmation, permission/enforcement/restoration evidence and a later new quiet
window remain pending. No acceptance comparison or live launch gate is closed.
