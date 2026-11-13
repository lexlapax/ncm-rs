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
command/image/environment/affinity and parent relationship. Parent stat and
relationship are checked again after observation. The caller still supplies the
coordinator command and snapshot digest: neither declaration's origin is
independently authenticated here. The helper does not execute a role, send a
signal, change clocks or claim launch readiness, safe ancestor cleanup topology,
continuous liveness or immutable mappings.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_role_binding -v
```

Four tests pass. Three use a private actual snapshot with controlled procfs stubs;
the fourth starts a real CPU0 coordinator and a separate taskset-launched `-S`
Python role, verifies their actual command/image/environment/affinity/parent
transport, and exits without executing a diagnostic. No GPU or policy action is
involved. Negative cases cover snapshot digest/privacy, parent command, environment,
affinity and nested ancestry. The future entry must authenticate the coordinator
command and snapshot source, recheck current files, bind the role dispatch to this
check, and independently verify permissions/restoration and safe ancestry before
any live action. Sampled identity is not a complete lifecycle guarantee.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 424 reporting tests. Local log: `/tmp/ncm-phase2-clock-role-binding-check.log`.

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

Even a successful root fake test would establish only those synthetic owned-child
transactions. It would not establish safe wiring to the unprivileged coordinator,
real CUDA cleanup, GPU clock enforcement/restoration or readiness. The user should
not run the real lock/reset directly: the approved protocol requires an armed and
reviewed restoration supervisor and a fresh quiet window first.
