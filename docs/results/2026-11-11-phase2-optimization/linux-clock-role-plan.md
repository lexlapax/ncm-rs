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
No password was requested in chat.

```sh
sudo /usr/bin/python3 -I -S -c 'import os,stat,hashlib; p="/tmp/ncm-root-fake-restoration-v2.py"; f=os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),"rb"); s=os.fstat(f.fileno()); assert stat.S_ISREG(s.st_mode) and s.st_size<=1048576; b=f.read(1048577); f.close(); assert len(b)==s.st_size and hashlib.sha256(b).hexdigest()=="710a7b0ff4911567054f4d9dbf9ca3d07a528aa61e172a83ed26e11b93d27ccc"; exec(compile(b,p,"exec"))' --privileged-test
```

The user's first invocation returned `Choose exactly --self-test or
--privileged-test`; no fake case ran. A corrected checksum-bound command sets
the script arguments inside the verified Python launcher. The user ran the
corrected short-path command below.
The short-path copy has the same SHA-256 as v2:

```sh
sudo python3 -I -S -c 'import os,sys,hashlib;p="/tmp/ncm-root-v2.py";f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);b=os.read(f,65536);os.close(f);assert hashlib.sha256(b).hexdigest()=="710a7b0ff4911567054f4d9dbf9ca3d07a528aa61e172a83ed26e11b93d27ccc";sys.argv=[p,"--privileged-test"];exec(compile(b,p,"exec"))'
```

The user-reported output and root-owned (outside-sandbox UID 0, mode 644) retained
summary agree: the normal case completed, the deadline case failed as expected,
and both recorded fake set/reset, successful reset submission, absent fake lock,
and complete owned cleanup. `/tmp/ncm-root-fake-restoration-v2-0.json` SHA-256:
`7c34e115a6201055e494e33124b5222d3c77eb6427dddedca3ee21f59678b523`.
No GPU or real policy command ran. This root fake test establishes only those
synthetic owned-child transactions. It does not establish safe wiring to the unprivileged coordinator,
real CUDA cleanup, GPU clock enforcement/restoration or readiness. The user should
not run the real lock/reset directly: the approved protocol requires an armed and
reviewed restoration supervisor and a fresh quiet window first.

### Potential terminal-owned process-group probe

A separate CPU-only exploratory script, `/tmp/ncm-sudo-tty-group-probe-v1.py`
(SHA-256 `cd00adf46b97c766d40f8d111137f8a5015dff3af3ef673676eb9e2f46c6801c`),
was run by the user in their terminal. It starts only `sudo -n
/usr/bin/sleep 3` in a new process group while retaining the parent session, then
observes bounded termination. The sudo process remained alive for 300 ms, received
a group SIGTERM, and exited with return code -15; the original group was absent
after the wait. The summary SHA-256 is
`6e58d0b5ad2be31909cd33ab1077c38789a805eda9e5b36e83fdad6a4fd72645`.
There was no root-child identity or readiness marker, so this does not establish
that a privileged child started or that an escaped child was cleaned up. It is not
a proposed change to the approved diagnostic.
A local v0 harness run without a terminal exited immediately with `sudo: a password
is required`; no privileged sleep ran. Its script and summary SHA-256 values are
`9e2b6c7bef44966923de21f01a616f4437efc954a39973a038c215b6849231d4` and
`43a1d55ccc48b0c0a0946de7814aaccbed72725e08f259907ecf187d1803f5af`.
The agent made no GPU access, credential or sudo configuration change in this probe. Any replacement
for the existing session-based capture requires reviewed design and comprehensive
fake cleanup tests; it is not authorized for a live clock operation by this probe.

The user ran a stronger CPU-only marker probe. Script
`/tmp/ncm-sudo-tty-group-marker-v1.py` has SHA-256
`6ba33b6ea1384bbff4a8fd12eee4c59a28b0d76cd1d64e9691a6d14ee0444865`.
It confirmed root-child readiness (`READY_UID_0`) and the fake child recorded
`signal-15` with effective UID 0 after the group signal; sudo returned 143 and
the original group was absent after waiting. Retained summary SHA-256 is
`1d73536b7ff5c79f3f566f918c7afa27d62b70009bedcbae8da88e7fa2c8243e`.
This establishes signal delivery to that fake child only, not complete privileged
cleanup, parent-loss behavior, hardware restoration or launch readiness. Its local
no-TTY v0 failed authentication before starting
a root child; script and summary hashes are
`089e70426381dad55ade550a97436cf159c4437589a9ad8a0790a25c65c16447` and
`6ba7b26b182a6ba02f45f4ddb60ea7ab2c8c4c84a10e0c303e2f935a615a709e`.
No password is requested in chat.

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

### Unprivileged TTY topology observation

One local CPU-only pseudo-terminal probe, `/tmp/ncm-clock-tty-topology-v1.py`
(SHA-256 `3f75268d6a6cb8cef47e0f0e03251bcfc161d3790f2a36b28cda04e66f54a827`),
created an owned coordinator with a controlling PTY and launched two ordinary
Python children with stdin set to `/dev/null`. The child started in a separate
process group within the coordinator's session could open `/dev/tty`; the child
started in a new session could not. Both reported non-TTY stdin. The one probe
ran as `python3 -I -S /tmp/ncm-clock-tty-topology-v1.py`; its JSON was observed in
the session, not retained as a raw log. It invoked neither sudo nor GPU commands.

This demonstrates only Linux TTY topology for that test. It does not prove sudo
cache behavior, root-child cleanup, signal delivery, privilege, a safe live role
architecture or permission to change the approved session-based capture. The
same-terminal sudo/sleep probe is a separate, limited experiment above.

### Local sudo manual review

The installed `sudo(8)` manual (sudo 1.9.17p2) describes credential caching as
per-terminal by default, and documents that sudo can relay signals to a child
but cannot relay `SIGKILL`. It also states user-requested `sudo -T` command timeouts
require explicit policy support. This is a documentation review, not proof of the
machine's effective timestamp, pty or timeout policy. In particular, the current
capture primitive's final group `SIGKILL` cannot by itself be assumed to terminate
an independently privileged command. Additional bounded cleanup and parent-loss tests are necessary before considering any
privileged integration; no GPU policy command was run.

### Credential lifetime and supervisor topology

The successful fake root-child signal test does not by itself make two separate `sudo -n nvidia-smi` calls safe for the fixed diagnostic.
The local sudo manual describes a default five-minute credential cache, while
the restoration contract allows a lock interval up to 600 seconds. The effective
local timeout has not been established. A cache that expires between set and
reset could make the reset require interaction; do not rely on a prior `sudo -v`
for the full lock lifetime. The user has not authorized the agent to change
sudoers or timestamp policy, the diagnostic limits or the restoration guarantee.

A privileged supervisor authenticated once and kept alive through reset might
avoid that particular expiry problem, but its parent/child ownership, monitoring,
user identity, workload environment, parent-loss cleanup and receipt bindings would
need a concrete reviewed design and fake tests. The current coordinator and
finalizer require specific direct-child relationships; inserting a sudo monitor
or an independently launched root process cannot be assumed to satisfy them.
The current `clock_worker_handoff.validate_handoff` also requires the private
restoration directory to be owned by the worker effective UID, and reads its
parent process metadata. A root restorer and a dropped-privilege worker cannot
be assumed compatible with those checks or Linux process-access rules; any
change needs explicit identity and provenance guarantees, not a relaxed check.
A fresh CPU-only run of `umask 077 && python3 -m unittest
test_clock_worker_handoff.ClockHandoffTests.test_default_budget_handoff_preserves_deadline_and_all_eighteen_real_observations`
from `tools/bench` passed once in 2.392 seconds. It does not resolve the retained
intermittent observer mismatches or exercise a root supervisor. Five further
consecutive executions of the same test passed (2.402–2.419 seconds each);
private JSON log `/tmp/ncm-clock-handoff-diagnostic-20261113.json` SHA-256
`840a5a17f728d330754101770bb4376804aa375eea01a5a13233ba3b5319bac8`.
A further bounded batch of twenty consecutive runs passed; its private log
`/tmp/ncm-clock-handoff-diagnostic-20261113-v2.json` has SHA-256
`fd00ca4429f16d30f3ad5186a4a43b491cdff551d4ca7402119b80840ea309b7`.
These passes do not explain or discard the retained failures.

A CPU-only dropped-user access probe has been prepared and requested from the
user. `/tmp/ncm-root-worker-access-v1.py` SHA-256
`0413e7a1bb3452fc29bcbfbdffc0eca3a8a2cd1a584fc5ff28795181569c9`
checks access to a fake root-owned obligation and its parent process metadata.
The user reported root UID 0 and dropped worker UID/GID 1000: the parent cmdline
and children were readable, while its environment and executable stat failed
with `PermissionError:13`; the root-owned obligation also failed with that error
and its directory owner did not match the worker. A transcription of the user
output is retained at `/tmp/ncm-root-worker-access-v1-user-output.json` (SHA-256
`fb1946c41f4a87bba3069d7f443cc55db95b5438f0b8420caff48b96a2df35c8`);
it is user-reported output, not a root-signed receipt. An unprivileged invocation
rejected itself as intended. The probe performed no GPU or real policy action
and does not implement the supervisor.

The role binder also reads private launch evidence through `Evidence`, which
requires files and directories owned by the inspecting real UID. A privileged
reader cannot simply consume the ordinary user's private launch snapshot under
that contract. The coordinator captures a direct restorer child and finalization
uses matching lifecycle and receipt identities; an intervening sudo monitor
therefore needs explicit treatment, not an assumed PID substitution. An eventual
design must bind trusted reviewed bytes before executing as root, ensure fake
set/reset remains owned by the surviving privileged restorer, preserve a
normal-user benchmark identity (including groups, capabilities and output
ownership), and authenticate every cross-boundary handoff. It must test parent
loss, signal/timeout cleanup and final receipts before any real policy action.
These are design requirements, not implemented mechanisms or a claim that a
privileged bootstrap can safely import a user-writable tree.
Nothing here selects an architecture or authorizes a real clock change.


### Sampled worker credentials

`tools/bench/clock_unprivileged_identity.py` supplies an import-only, fail-closed
credential check for a future unprivileged worker. It requires a single-threaded
Linux process and explicit non-root UID, primary GID and supplementary-group
multiset. It checks all four procfs UID/GID values, corroborates real/effective/
saved IDs and groups with process APIs, and requires zero inheritable, permitted,
effective and ambient capabilities. The bounding capability set is not active
and is not required to be empty. No rights are changed. Seven focused tests pass,
including a live read of the sandbox process, malformed data, active capabilities
and mismatched identities. The first focused run failed because the sandbox
exposes repeated mapped group IDs; matching now retains group multiplicity and
the subsequent seven tests pass.

This does not authenticate the caller's expected identity, drop privileges,
inspect future descendants, prevent later credential changes, or prove launch
readiness. It is not yet wired to the worker or a privileged entry. Identity
sampling cannot replace trusted code provenance, handoff/parent binding,
restoration and the remaining Linux and Apple hardware gates. No GPU action ran.

The full `umask 077 && CARGO_BUILD_JOBS=4 ./tools/check.sh` run outside the
sandbox failed one of 452 reporting tests: the real CPU-only handoff fixture in
`test_validate_clock_restoration` stopped at job 15 with
`live child command mismatch: collector_command` and zero accepted samples for
that job; fake reset succeeded. This identifies equality with the collector's
command in that sample, not why it occurred or whether the earlier mismatches
were identical. It must not be promoted to a valid observation. The private log
`/tmp/ncm-worker-identity-check-unsandboxed.log` has SHA-256
`55932ebdfb38c76645cc5b8117b0544dd6e9fa52be808d33797dbd2d508b6fe2`.
The check started before the final additional thread test and Linux-only skip
were added; all seven focused identity tests subsequently passed. The initial
sandboxed check stopped earlier because sccache returned operation not permitted
(log `/tmp/ncm-worker-identity-check.log`, SHA-256
`12867ac490c15de556c59a51d9211b485d426708fe382f10d2af372a28055227`).


### Collector-command investigation

A separate CPU-only stress harness `/tmp/ncm-observer-exec-probe-v1.py` (SHA-256
`e85cb969e6f5d8c507181851bb98e314f866a95f82e933d09cc29c6232fae7fb`)
ran the actual 18-job fake handoff fixture, adding only failure-time diagnostics.
Its 28th attempt failed at job 11 with `collector_command`, before any accepted
sample for that job. The immediate follow-up still matched the collector's
command and executable; `/proc/<pid>/syscall` reported `running`, not a syscall
number. Fake reset succeeded and the reaper role was restored. The retained
private results `/tmp/ncm-observer-exec-probe-v1.json` have SHA-256
`d2f928af46cb69462dbcda8e740ea0b5075e37328735b0223f7a4882a79ec46e`.
Instrumentation reads only test-owned fake children; no GPU or policy action ran.

The installed Python 3.14 subprocess source uses a fork/exec error pipe for the
relevant `start_new_session=True` path. In the inspected [Linux exec source at a
fixed upstream revision](https://code.googlesource.com/linux/torvalds/linux/+/5d144c294ae21c1bfb275ba06ec73c2d1ab7cf92/fs/exec.c),
close-on-exec processing precedes installation of the new memory image. This
makes a transition race a plausible explanation, not proof of this machine's
specific event or of earlier failures; the inspected upstream revision was not
verified as the running Arch kernel source. The follow-up syscall result does
not identify an exec syscall. No acceptance, timing, observation or failure
rule changed; command mismatches remain immediate failures.

A new focused regression injects the collector command into sampled child
command reads, verifies an immediate failed receipt, zero accepted samples and
`workload_not_observed`. `umask 077 && PYTHONPATH=tools/bench python3 -m unittest
-v test_observe_readonly_process` passed all 13 tests; private log
`/tmp/ncm-observer-exec-focused-v1.log` SHA-256
`94a6305bd35700c1d3b6daefb7b77a5908e211a36452a5ab306ea1e5ef599911`.
This focused pass does not resolve the retained full-check failure, establish a
safe startup barrier, or make the diagnostic ready to run.
