"""Read-only live role identity checks; no diagnostic command or policy action.

Caller supplies reviewed coordinator argv and a private snapshot hash. Checks are
sampled and do not authenticate the caller, prove ancestor cleanup topology, or
permit collection. No role handler is implemented here.
"""
import hashlib
import os
from pathlib import Path
import signal
import sys
import threading

from clock_evidence import Evidence
from clock_role_plan import role_plan
from finalize_clock_diagnostic import parsed
from observe_readonly_process import environment, process_stat, read_proc
from readonly_trace_preflight import require
from supervised_worker import direct_children, subreaper

ROLES = ('restorer','worker','observer-supervisor','telemetry')


def inspect(role, *, launch, snapshot_sha256, coordinator_pid, coordinator_argv, gpu_uuid):
    """Check a single already-started internal role against supplied declarations.

    Requires the role's actual parent and coordinator to remain live and unchanged
    across sampled procfs checks. The launch entry must authenticate the coordinator
    argv and frozen snapshot itself; this helper cannot prove safe earlier ancestry,
    continuous identity, immutable mapped bytes or exclusive later reaping.
    """
    require(role in ROLES and sys.platform == 'linux' and sys.flags.no_site == 1
            and threading.current_thread() is threading.main_thread() and threading.active_count() == 1
            and len(list(Path('/proc/self/task').iterdir())) == 1
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL and not direct_children()
            and subreaper() == 0, 'fresh single-threaded Linux -S role required')
    require(type(coordinator_pid) is int and coordinator_pid > 1
            and isinstance(coordinator_argv, list) and coordinator_argv
            and all(isinstance(a, str) and a and '\0' not in a for a in coordinator_argv)
            and Path(coordinator_argv[0]).is_absolute(), 'explicit coordinator declaration required')
    launch = Path(launch)
    with Evidence(launch) as evidence:
        raw = evidence.raw('snapshot.json')
        require(isinstance(snapshot_sha256, str) and hashlib.sha256(raw).hexdigest() == snapshot_sha256,
                'private snapshot digest differs')
        snapshot = parsed(raw)
        plan = role_plan(snapshot, snapshot_path=launch/'snapshot.json', snapshot_sha256=snapshot_sha256,
                         launch=launch, coordinator_pid=coordinator_pid, gpu_uuid=gpu_uuid)
        evidence.verify_unchanged()
    roles = plan['roles']
    expected_self = plan['observer_argv'] if role == 'observer-supervisor' else roles[
        'telemetry_worker' if role == 'telemetry' else role]['argv']
    parent_pid = os.getppid()
    if role in ('restorer','observer-supervisor'):
        require(parent_pid == coordinator_pid, 'role is not a coordinator child')
        expected_parent = coordinator_argv
    else:
        require(parent_pid != coordinator_pid and parent_pid > 1, 'dedicated role parent required')
        expected_parent = (roles['restorer']['argv'] if role == 'worker' else plan['observer_argv'])[3:]
    expected_image = snapshot['utilities'][0]
    image_key = expected_image['device'], expected_image['inode']
    expected_env = {k.encode(): v.encode() for k,v in plan['environment'].items()}
    require(dict(os.environ) == plan['environment'] and os.getcwd() == snapshot['project_root']
            and os.sched_getaffinity(0) == {0}, 'current role environment/cwd/affinity differs')
    self_image = os.stat('/proc/self/exe')
    require((self_image.st_dev,self_image.st_ino) == image_key
            and read_proc('/proc/self/cmdline',65536) == b'\0'.join(x.encode() for x in expected_self[3:])+b'\0',
            'current role image/command differs')

    def check_process(pid, argv):
        before = process_stat(read_proc(f'/proc/{pid}/stat',65536))
        image = os.stat(f'/proc/{pid}/exe')
        require(before[0] == pid and (image.st_dev,image.st_ino) == image_key
                and os.sched_getaffinity(pid) == {0}
                and environment(read_proc(f'/proc/{pid}/environ',65536)) == expected_env
                and read_proc(f'/proc/{pid}/cmdline',65536) == b'\0'.join(x.encode() for x in argv)+b'\0',
                'declared parent/coordinator identity differs')
        return before

    parent_before = check_process(parent_pid, expected_parent)
    if parent_pid != coordinator_pid:
        require(parent_before[1] == coordinator_pid, 'role parent is not coordinator child')
        coordinator_before = check_process(coordinator_pid, coordinator_argv)
        require(process_stat(read_proc(f'/proc/{coordinator_pid}/stat',65536)) == coordinator_before,
                'coordinator changed during observation')
    require(os.getppid() == parent_pid
            and process_stat(read_proc(f'/proc/{parent_pid}/stat',65536)) == parent_before,
            'role parent changed during observation')
    return {'schema_version':1, 'role':role, 'pid':os.getpid(), 'parent_pid':parent_pid,
            'coordinator_pid':coordinator_pid, 'snapshot_sha256':snapshot_sha256,
            'sampled_role_identity_verified':True, 'coordinator_declaration_authenticated':False,
            'safe_ancestor_topology_verified':False, 'continuous_liveness_verified':False,
            'launch_ready':False,
            'scope':'Sampled procfs command/image/environment/affinity and parent bindings only. '
                    'Supplied coordinator declaration and snapshot origin are not authenticated; '
                    'not safe ancestor topology, continuous liveness or launch authorization.'}
