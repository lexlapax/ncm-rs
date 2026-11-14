"""Bounded codec worker dispatch; import-only, not a launch entry.

The caller must authenticate coordinator/snapshot declarations and fresh availability.
This module contains no clock policy command; the separately owned restorer must
already have armed its obligation and the entry must establish safe privilege/topology.
"""
import os
from pathlib import Path

from clock_codec_sequence import run_sequence
from clock_drop_privileges import drop_and_verify
from clock_preflight import read_private
from clock_process_observation import process_factory
from clock_role_binding import inspect as inspect_binding
from clock_role_plan import role_plan
from clock_worker_handoff import validate_handoff
from clock_unprivileged_identity import verify as verify_unprivileged
from compare_clock_snapshots import validate as validate_snapshot
from readonly_trace_preflight import require


def run(*, launch, snapshot_sha256, coordinator_pid, coordinator_argv, gpu_uuid,
        quiet_window_confirmed, privilege_target=None):
    """Bind the owned worker to the original armed deadline and observed jobs.

    The caller's quiet-window and coordinator declarations remain unauthenticated;
    this wrapper does not prove clock state, hardware restoration, safe ancestor
    topology or collection authorization. Any failure must become nonzero in the
    future role entry, and no workload is retried here. If started as root, the
    caller must supply an independently authenticated target UID/GID/groups. The
    armed handoff is checked before the irreversible drop, and no sequence starts
    until the drop succeeds. This wrapper does not authenticate the target.
    """
    require(quiet_window_confirmed is True, 'fresh quiet-window declaration required')
    uids, gids = os.getresuid(), os.getresgid()
    privileged = uids == (0, 0, 0) and gids == (0, 0, 0)
    require(privileged or (len(set(uids)) == len(set(gids)) == 1
                           and uids[0] > 0 and gids[0] > 0),
            'mixed or root credentials require a dedicated root worker')
    require((privilege_target is not None) == privileged,
            'root worker requires an explicit privilege target')
    if privileged:
        require(isinstance(privilege_target, dict) and set(privilege_target) == {'uid', 'gid', 'groups'}
                and isinstance(privilege_target['groups'], (list, tuple)),
                'explicit UID/GID/group target required')
        privilege_target = dict(privilege_target, groups=tuple(privilege_target['groups']))
    binding = inspect_binding('worker',launch=launch,snapshot_sha256=snapshot_sha256,
        coordinator_pid=coordinator_pid,coordinator_argv=coordinator_argv,gpu_uuid=gpu_uuid)
    require(binding['sampled_role_identity_verified'] is True and binding['role'] == 'worker'
            and binding['launch_ready'] is False, 'sampled worker role binding required')
    launch = Path(launch)
    snapshot = validate_snapshot(read_private(launch/'snapshot.json',snapshot_sha256))
    plan = role_plan(snapshot,snapshot_path=launch/'snapshot.json',snapshot_sha256=snapshot_sha256,
                     launch=launch,coordinator_pid=coordinator_pid,gpu_uuid=gpu_uuid)
    root=Path(snapshot['project_root']); collection=Path(snapshot['collection_root'])
    contract=snapshot['inventory']['library_contract']; taskset=contract['taskset']['path']
    commands={name:plan['roles'][name]['argv'] for name in ('set','worker','reset')}
    parent_pid=os.getppid()
    deadline=validate_handoff(collection/'restoration/obligation.json',parent_pid=parent_pid,
        parent_argv=plan['roles']['restorer']['argv'][3:],commands=commands,
        role_image=snapshot['utilities'][0],taskset_path=taskset,cwd=root,env=plan['environment'])
    if privileged:
        uid, gid, groups = (privilege_target[key] for key in ('uid', 'gid', 'groups'))
        dropped = drop_and_verify(uid, gid, groups)
        require(isinstance(dropped, dict) and dropped.get('sampled_unprivileged_identity_verified') is True
                and dropped.get('no_new_privs_verified') is True and dropped.get('launch_ready') is False
                and os.getresuid() == (uid, uid, uid) and os.getresgid() == (gid, gid, gid)
                and tuple(sorted(os.getgroups())) == tuple(sorted(groups)), 'worker privilege drop unverified')
    else:
        uid, gid, groups = uids[0], gids[0], tuple(os.getgroups())
    identity = verify_unprivileged(uid, gid, groups)
    require(isinstance(identity, dict) and identity.get('sampled_unprivileged_identity_verified') is True
            and identity.get('launch_ready') is False, 'worker identity unverified')
    injection_path=str(root/'.reference/phase2-cupti/libcupti_trace_injection.so')
    injection=[v for v in snapshot['native']['file_identities'] if v['path']==injection_path]
    require(len(injection)==1, 'one frozen injection identity required')
    factory=process_factory(snapshot['jobs'],env=plan['environment'],injection=injection[0],
        **{key:contract[key] for key in ('workloads','taskset','libraries','known_libraries')})
    return run_sequence(root/'.reference/phase2-optimization-baseline',
        root/'.reference/phase2-optimization-baseline-bin',collection/'sequence',root=root,
        env=plan['environment'],work_deadline=deadline,cancel_check=lambda:os.getppid()!=parent_pid,
        process_observer_factory=factory,taskset=taskset)
