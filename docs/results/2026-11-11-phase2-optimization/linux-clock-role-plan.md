# Clock diagnostic role plan

`clock_role_plan.role_plan` constructs an explicit set of six finalizer role
receipt declarations, the sibling observer-supervisor command, and the GPU
telemetry command from a supplied non-launch-ready snapshot, GPU UUID and
coordinator PID. It is pure command construction: it runs no command, inspects no
process, reads no files and does not authorize clock changes. The actual entry
and live role handlers have not been implemented.

The plan fixes coordinator-related roles on CPU0; the CPU sampler starts on CPUs
0 and 2 and performs its existing transition to CPU0. The restorer has the tested
660-second outer limit and 60-second termination grace; set/reset each have the
existing 10-second limit, the worker 600 seconds with its separate 590-second
absolute work deadline enforced by restoration, the observer worker 950 seconds,
and CPU telemetry 930 seconds. All capture/report limits match existing helpers.
The plan includes one requested 1710 MHz lock and one reset command as data;
only the existing restoration supervisor may later perform the allowed cleanup-only
retry. It does not infer supported clocks, permissions or successful restoration.
Each role carries the GPU UUID and the fixed private `snapshot.json` path under
the launch directory, enabling future handlers to recompute the same plan from
the snapshot digest and explicit arguments. This convention is not proof that
the private copy is authentic.

The file snapshot must include the future entry in its tool inventory; existing
historical snapshot v1 predates that entry and cannot satisfy this requirement.
A future entry must independently authenticate the snapshot digest and origin,
validate role commands, parents/images/environments and safe sibling topology,
repeat current preflight, and establish GPU identity, permissions, restoration
and a fresh quiet window. The plan's own flags deliberately remain false. It does
not prove safe live ancestry, actual reaping, telemetry coverage, hardware state,
complete preparation or performance acceptance.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_role_plan test_clock_control_evidence test_clock_diagnostic_coordinator -v
```

Nineteen focused tests pass, including command budgets, CPU affinity declarations,
isolation, scope/path/environment/UUID failures, existing receipt binding and
synthetic coordinator cleanup behavior. No clock, GPU, workload or telemetry
collection was run for this component. Focused-test log:
`/tmp/ncm-phase2-clock-role-plan-focused.log`.
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 420 reporting tests; its local log is
`/tmp/ncm-phase2-clock-role-plan-check.log`. That run began before the private
snapshot path and UUID argument refinement, which then passed the two focused
`test_clock_role_plan` tests separately.

## Sampled role binding

`clock_role_binding.inspect` reads a private `snapshot.json` under the launch
folder, verifies its supplied digest, recomputes the fixed plan, and checks the
current role's command/image/environment/CPU0 affinity. It samples its live
parent and, for worker/telemetry grandchildren, the declared coordinator's
command/image/environment/cwd/affinity and parent relationship. Parent stat and
relationship are checked again after observation. The caller still supplies the
coordinator command and snapshot digest: neither declaration's origin is
independently authenticated here. The helper does not execute a role, send a
signal, change clocks or claim launch readiness, safe ancestor cleanup topology,
continuous liveness or immutable mappings.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_role_binding -v
```

Five tests pass. Four use a private actual snapshot with controlled procfs stubs;
the fourth starts a real CPU0 coordinator and a separate taskset-launched `-S`
Python role, verifies their actual command/image/environment/affinity/parent
transport, and exits without executing a diagnostic. No GPU or policy action is
involved. Negative cases cover snapshot digest/privacy, parent command, environment,
affinity and nested ancestry. The future entry must authenticate the coordinator
command and snapshot source, recheck current files, bind the role dispatch to this
check, and independently verify permissions/restoration and safe ancestry before
any live action. Sampled identity is not a complete lifecycle guarantee.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 424 reporting tests initially and 438 after the cwd addition. Local
logs: `/tmp/ncm-phase2-clock-role-binding-check.log` and
`/tmp/ncm-phase2-role-cwd-check.log`.

## Noninteractive permission investigation

On 2026-11-13, a read-only `sudo -n -l` check for each exact proposed lock and
reset command, using the **historical** GPU UUID
`GPU-180283f7-eafd-6501-7e06-eef40d304c41`, returned exit zero and displayed the
matching command outside the sandbox. The first lock-policy listing inside the
sandbox failed because sudo was blocked by `no_new_privs`; it did not execute a
GPU command. Outside the sandbox, the effective UID was 1000 and the permitted
and effective capability masks were zero. These observations do not prove the
current GPU UUID, direct unprivileged lock permission, actual enforcement,
credential availability at a later launch, or restoration.

```sh
sudo -n -l /usr/bin/nvidia-smi -i GPU-180283f7-eafd-6501-7e06-eef40d304c41 --lock-gpu-clocks=1710,1710
sudo -n -l /usr/bin/nvidia-smi -i GPU-180283f7-eafd-6501-7e06-eef40d304c41 --reset-gpu-clocks
```

The current role plan invokes `nvidia-smi` directly; it does not invoke sudo.
Do not add sudo or attempt clock changes until the supervisor can safely bound,
cancel and reap the privileged process tree and the final command/environment
bindings are reviewed and tested. A read-only `sudo -n -l /usr/bin/sleep 1` also
returned exit zero, suggesting a harmless fake-command path for further cleanup
research. These were policy listings only; no privileged command was executed,
no credentials or configuration changed by the agent, and no GPU query, clock or
benchmark action occurred during these policy listings.

### Harmless privileged-child probe

One bounded CPU-only test attempted `sudo -n /usr/bin/sleep 3` under the existing
owned-child capture, with a 0.3-second timeout. Sudo exited in 0.0044 seconds with
`sudo: a password is required`; the intended privileged sleep did not start. The
receipt reports a nonzero exit, direct-child reaping and capture EOF, but this is
**not** a privileged descendant cleanup test. Detailed `sudo -n -ll` listings for
both NVIDIA commands matched an `ALL` rule; that listing alone does not establish
password-free execution. No GPU or policy command ran. The user requested a sudo
prompt and ran `sudo -v` in their own terminal without sending a password or
changing clocks. The agent's subsequent `sudo -n /usr/bin/true` still required a
password. At the user's terminal, a second bounded fake test after `sudo -v` also
failed immediately with `sudo: a password is required`; no privileged sleep ran.
The detached child therefore did not inherit usable authorization, and neither
test establishes privileged-child cleanup. A reviewed interactive privilege and
cleanup design remains necessary; do not silently alter sudo configuration.
Role planning, authorization and safe privileged cleanup still need integration.

Private evidence: `/tmp/ncm-clock-sudo-owned-cleanup-v1.py` SHA-256
`bacf4a68efad2de68dc7328bcc5a541694efed94b5762d0966a16c517a9fe1bc`,
`/tmp/ncm-clock-sudo-owned-cleanup-v1/receipt.json` SHA-256
`64d072c08b5c65331e06f0ec84dc307999728cde62698c719dce1bfb2bb86841`,
`capture.txt` SHA-256 `9b3a3b7a78f239c4473bb193476b1a860c91f1c2b4fab9c7639eceea414a60fb`,
and `summary.json` SHA-256
`a2109f6be245d7a2f38273cdbc92318d26f66fce4747789018ea18a6873607a3` in the same directory.

The user-run v2 script `/tmp/ncm-clock-sudo-owned-cleanup-v2.py` SHA-256 is
`b52b57d54d398b0d9ac18b8832b3a54b58d80270ad8cb6669152fceef641e5fb`;
its private `receipt.json`, `capture.txt` and `summary.json` under
`/tmp/ncm-clock-sudo-owned-cleanup-v2/` have SHA-256 values
`7ed5b3646d307cdadbce7b56c32178d453f5dbf0d2d058811ef26ccb2b61ec95`,
`9b3a3b7a78f239c4473bb193476b1a860c91f1c2b4fab9c7639eceea414a60fb`, and
`d0886f652b5046c71863c063c509a0d11c740f45fff650c4efe3fa9a83f54429` respectively.

### Read-only GPU identity and supported clocks

A later read-only query outside the sandbox reported the same historical UUID,
PCI `00000000:01:00.0` and NVIDIA GeForce RTX 3060 Ti. The corresponding
inside-sandbox identity query failed to communicate with the driver. The
outside-sandbox supported-graphics-clock query returned 114 rows, including
`1710 MHz`; retained `/tmp/ncm-phase2-supported-clocks-v1.csv` SHA-256
`b90cfe2abbb98c11366c42bf2dfaebfdb140444476a1f00a10805cc99658441f`.
The utility describes this listing as application-clock combinations. It does
not prove that the proposed GPU lock is permitted, accepted or enforced, and the
identity must be rebound during the final fresh preflight. No clock was set.

```sh
/usr/bin/nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader
/usr/bin/nvidia-smi -i GPU-180283f7-eafd-6501-7e06-eef40d304c41 --query-supported-clocks=graphics --format=csv,noheader > /tmp/ncm-phase2-supported-clocks-v1.csv
```

### User-run CPU-only privileged supervisor test

At the user's request for a sudo command, a checksum-bound standalone test was
prepared at `/tmp/ncm-root-fake-restoration-v2.py` (SHA-256
`710a7b0ff4911567054f4d9dbf9ca3d07a528aa61e172a83ed26e11b93d27ccc`).
It copies four hash-pinned reviewed supervisor helpers into a private temporary
directory before import, then uses only fake local set/reset files and Python sleep
workers. Its ordinary-user `--self-test` passed both normal and deadline cases;
the initial v1 self-test exposed an incorrect parent PID and was corrected before
v2. The user has been given an `sudo /usr/bin/python3 -I -S -c ...` command that
reads and verifies those exact script bytes before executing `--privileged-test`.
No password was requested in chat. Its result is pending.

```sh
sudo /usr/bin/python3 -I -S -c 'import os,stat,hashlib; p="/tmp/ncm-root-fake-restoration-v2.py"; f=os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),"rb"); s=os.fstat(f.fileno()); assert stat.S_ISREG(s.st_mode) and s.st_size<=1048576; b=f.read(1048577); f.close(); assert len(b)==s.st_size and hashlib.sha256(b).hexdigest()=="710a7b0ff4911567054f4d9dbf9ca3d07a528aa61e172a83ed26e11b93d27ccc"; exec(compile(b,p,"exec"))' --privileged-test
```

The user's first invocation returned `Choose exactly --self-test or
--privileged-test`; no fake case ran. A corrected checksum-bound command sets
the script arguments inside the verified Python launcher. Its output is pending.
The short-path copy has the same SHA-256 as v2:

```sh
sudo python3 -I -S -c 'import os,sys,hashlib;p="/tmp/ncm-root-v2.py";f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);b=os.read(f,65536);os.close(f);assert hashlib.sha256(b).hexdigest()=="710a7b0ff4911567054f4d9dbf9ca3d07a528aa61e172a83ed26e11b93d27ccc";sys.argv=[p,"--privileged-test"];exec(compile(b,p,"exec"))'
```

Even a successful root fake test would establish only those synthetic owned-child
transactions. It would not establish safe wiring to the unprivileged coordinator,
real CUDA cleanup, GPU clock enforcement/restoration or readiness. The user should
not run the real lock/reset directly: the approved protocol requires an armed and
reviewed restoration supervisor and a fresh quiet window first.

### Potential terminal-owned process-group probe

A separate CPU-only exploratory script, `/tmp/ncm-sudo-tty-group-probe-v1.py`
(SHA-256 `cd00adf46b97c766d40f8d111137f8a5015dff3af3ef673676eb9e2f46c6801c`),
was prepared but not run in the user's terminal. It would start only `sudo -n
/usr/bin/sleep 3` in a new process group while retaining the parent session, then
observe bounded termination. This is a prospective fake test, not a proposed change
to the approved diagnostic or proof that privileged children can be cleaned up.
A local v0 harness run without a terminal exited immediately with `sudo: a password
is required`; no privileged sleep ran. Its script and summary SHA-256 values are
`9e2b6c7bef44966923de21f01a616f4437efc954a39973a038c215b6849231d4` and
`43a1d55ccc48b0c0a0946de7814aaccbed72725e08f259907ecf187d1803f5af`.
The agent made no GPU access, credential or sudo configuration change in this probe. Any replacement
for the existing session-based capture requires reviewed design and comprehensive
fake cleanup tests; it is not authorized for a live clock operation by this probe.

## Observer branch role dispatch

`clock_observer_roles.run` is an import-only wrapper for the observer-supervisor
and telemetry roles. It requires a caller-declared fresh quiet window, checks the
sampled live role binding and private snapshot digest, recomputes the fixed role
plan, binds the CPU sampler source digest and delegates to the existing separate
observer supervisor or telemetry collector. It contains no clock set/reset or
workload command. Its caller-supplied quiet-window and coordinator declarations
are **not authenticated** by this wrapper; only a future entry may supply them
after full preflight and user confirmation. It does not prove safe ancestor
cleanup topology or authorize an actual collection.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_observer_roles test_clock_role_binding test_clock_role_plan -v
```

Ten focused tests pass. Three new wrapper tests stub both observer actions and
cover the exact plan, missing quiet declaration, failed role binding and changed
snapshot; no telemetry or GPU command ran in those tests. Existing tests include
a real CPU-only role identity transport. The authenticated entry, privileged
restoration integration, real telemetry, and both platform gates remain pending.
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 441 reporting tests; local log: `/tmp/ncm-phase2-clock-observer-roles-check.log`.

A subsequent test-only follow-up runs a real CPU0 parent and observer-role
process through the actual snapshot/parent binding into a **stubbed** supervisor
dispatch, without starting telemetry or creating the collection directory. Nine
focused observer/binding tests pass. The full default check was not repeated for
this test-only addition; the preceding full run above remains the recorded check.

## Worker role dispatch

`clock_worker_role.run` is an import-only wrapper that requires a caller-declared
quiet window and sampled worker binding, re-reads and validates the private snapshot,
recomputes the role plan, and validates the live restorer handoff. It passes the
original armed absolute deadline to the existing bounded codec sequence and binds
all 18 frozen jobs to the process-observer factory. Parent loss feeds cancellation;
no clock policy command is executed by this wrapper. A future authenticated entry
must turn any failure into a nonzero worker exit and establish privilege, role
ancestry, fresh availability and restoration before collection. Supplied quiet and
coordinator declarations are not authorization by themselves.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_worker_role test_clock_observer_roles test_clock_role_binding -v
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_worker_role test_clock_worker_handoff test_clock_codec_sequence -v > /tmp/ncm-phase2-worker-role-focused.log 2>&1
```

The first command passed 11 tests; three new wrapper tests stub the handoff,
observer factory and worker sequence and make no workload or GPU calls. The second
ran 16 tests and failed the previously intermittent real CPU handoff fixture: job
12's owned-child observer reported `live child command mismatch`; fake reset and
cleanup completed. No timing, expectation, observer guard or limit changed. The
failure is retained in the local log; its cause remains unestablished and the
handoff is not declared reliable. No CUDA or real policy operation occurred.
An outside-sandbox `umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` subsequently
passed all default checks including 444 reporting tests; local log:
`/tmp/ncm-phase2-clock-worker-role-check.log`. That pass does not explain or erase
the preceding command-mismatch failure.

## Observer command-mismatch classification

The owned-child observer still aborts on every stable sampled command mismatch; no
mismatch is reclassified as a successful or transient sample. For diagnosis only,
it now records one of three fixed categories: the other declared phase's command,
the collector's own command, or unclassified. It reads the collector's own
`/proc/self/cmdline` only on mismatch and never serializes command or environment
bytes. This can distinguish future occurrences but does not retroactively identify
the already-retained job-10/job-12 failures or establish that either was an exec
race. The failure remains latched and all prior guards and sample limits remain.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_observe_readonly_process test_clock_process_observation test_verify_readonly_processes -v > /tmp/ncm-phase2-command-category-focused.log 2>&1
```

Twenty-nine focused tests pass, including fixed-category and no-secret assertions,
real CPU-only observation and existing postflight tests. No GPU, workload benchmark,
policy action or historical diagnostic rerun occurred.
A further targeted two-test run verified that an actual owned-child command
mismatch is still cancelled and retains only one of the fixed categories; it
overlapped briefly with the default check. No captured command bytes were added.

The default `umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` run passed 445
reporting tests. It began before the additional actual-mismatch assertion above;
the two updated focused tests passed during that run. Local full-check log:
`/tmp/ncm-phase2-command-category-check.log`.
