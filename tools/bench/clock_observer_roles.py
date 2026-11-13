"""Observer-branch role dispatch; import-only, not a launch entry.

The trusted caller must authenticate the coordinator and snapshot origin, establish
safe sibling topology and fresh availability, and separately supervise restoration.
This module contains no clock-setting/reset command and never starts a workload.
"""
from pathlib import Path
import re

from clock_preflight import read_private
from clock_role_binding import inspect as inspect_binding
from clock_role_plan import OBSERVER_WORKER_TIMEOUT, role_plan
from readonly_trace_preflight import require
from readonly_trace_telemetry import collect
from supervised_worker import supervise


def run(role, *, launch, snapshot_sha256, coordinator_pid, coordinator_argv, gpu_uuid,
        quiet_window_confirmed):
    """Run only the already-bound observer supervisor or its telemetry worker.

    The supplied quiet-window/coordinator declarations are trusted caller inputs,
    not independently authenticated authorization. Only the eventual entry may
    supply them after fresh preparation. This is not safe ancestor-topology proof.
    """
    require(role in ('observer-supervisor','telemetry') and quiet_window_confirmed is True,
            'explicit observer role and fresh quiet-window declaration required')
    binding = inspect_binding(role,launch=launch,snapshot_sha256=snapshot_sha256,
        coordinator_pid=coordinator_pid,coordinator_argv=coordinator_argv,gpu_uuid=gpu_uuid)
    require(binding['sampled_role_identity_verified'] is True and binding['role'] == role
            and binding['launch_ready'] is False, 'sampled observer role binding required')
    launch = Path(launch)
    snapshot = read_private(launch/'snapshot.json',snapshot_sha256)
    plan = role_plan(snapshot,snapshot_path=launch/'snapshot.json',snapshot_sha256=snapshot_sha256,
                     launch=launch,coordinator_pid=coordinator_pid,gpu_uuid=gpu_uuid)
    root = Path(snapshot['project_root']); collection = Path(snapshot['collection_root'])
    source_path = str(root/'tools/bench/sample_linux_cpu.py')
    matches = [v for v in snapshot['tool_identities'] if v['path'] == source_path]
    require(len(matches) == 1 and isinstance(matches[0].get('sha256'),str)
            and re.fullmatch('[0-9a-f]{64}',matches[0]['sha256']), 'frozen CPU sampler digest required')
    if role == 'observer-supervisor':
        return supervise(plan['roles']['telemetry_worker']['argv'],cwd=root,env=plan['environment'],
                         output_dir=collection/'observer-supervision',timeout_seconds=OBSERVER_WORKER_TIMEOUT,
                         parent_pid=coordinator_pid)
    return collect(plan['roles']['cpu']['argv'],plan['gpu_argv'],cwd=root,env=plan['environment'],
                   output_dir=collection/'telemetry',expected_cpu_source=matches[0]['sha256'],gpu_uuid=gpu_uuid)
