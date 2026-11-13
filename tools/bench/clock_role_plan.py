"""Build a fixed clock diagnostic role plan; never execute any command.

This is a supplied-snapshot declaration builder, not a live entry, policy check,
producer authentication or permission to change clocks. The future entry must
independently authenticate/recheck this plan and all role ancestry at launch.
"""
from copy import deepcopy
from pathlib import Path
import os
import re

from clock_codec_sequence import REPORT_BYTES
from clock_restoration_capture import CAPTURE_BYTES, RESTORATION_GRACE_SECONDS, SUPERVISOR_SECONDS
from readonly_trace_preflight import require
from readonly_trace_telemetry import CPU_TIMEOUT_SECONDS, commands as telemetry_commands
from verify_clock_files import canonical_spelling

OBSERVER_WORKER_TIMEOUT = 950
ENTRY_RELATIVE = 'tools/bench/clock_diagnostic_entry.py'


def role_plan(snapshot, *, snapshot_path, snapshot_sha256, launch, coordinator_pid, gpu_uuid):
    """Construct six finalizer role receipts plus the sibling observer launcher.

    The coordinator PID is fixed before submission; later role PIDs are not guessed.
    Command declarations have no authority to execute. A future entry must validate
    every role's parent/image/environment and enforce fresh preflight and readiness.
    """
    snapshot = deepcopy(snapshot)
    require(isinstance(snapshot, dict) and snapshot.get('kind') == 'clock-file-snapshot'
            and type(snapshot.get('schema_version')) is int and snapshot['schema_version'] == 1
            and all(snapshot.get(k) is False for k in ('launch_ready','collection_started',
                'gpu_identity_verified','hardware_policy_verified','permissions_verified',
                'restoration_verified','availability_confirmed')), 'non-launch-ready file snapshot required')
    root = canonical_spelling(snapshot['project_root'])
    collection = canonical_spelling(snapshot['collection_root'])
    snapshot_path = canonical_spelling(str(snapshot_path))
    launch = canonical_spelling(str(launch))
    require(root != Path('/') and snapshot_path == launch/'snapshot.json'
            and not collection.is_relative_to(launch) and not launch.is_relative_to(collection)
            and re.fullmatch('[0-9a-f]{64}', snapshot_sha256 or '')
            and type(coordinator_pid) is int and coordinator_pid > 1,
            'explicit disjoint paths, snapshot hash and coordinator required')
    identities = snapshot['tool_identities']; entry = str(root/ENTRY_RELATIVE)
    require(isinstance(identities, list) and sum(v.get('path') == entry for v in identities) == 1,
            'future entry must be frozen in tool inventory')
    utilities = snapshot['utilities']
    require(isinstance(utilities, list) and len(utilities) == 3, 'snapshot utility identities required')
    python, nvidia = utilities[0]['path'], utilities[1]['path']
    taskset = snapshot['inventory']['library_contract']['taskset']['path']
    for path in (python,nvidia,taskset): canonical_spelling(path)
    env = snapshot['environment']
    expected = {'PATH': os.defpath, 'LANG':'C', 'LC_ALL':'C',
                'PYTHONNOUSERSITE':'1', 'TOKENIZERS_PARALLELISM':'false'}
    require(isinstance(env, dict) and set(env) == set(expected) | {'HOME'}
            and all(env[k] == v for k,v in expected.items())
            and isinstance(env['HOME'], str) and Path(env['HOME']).is_absolute(),
            'isolated role environment required')
    canonical_spelling(env['HOME'])
    common = [taskset,'-c','0',python,'-S',entry]
    args = ['--snapshot',str(snapshot_path),'--snapshot-sha256',snapshot_sha256,
            '--launch',str(launch),'--coordinator-pid',str(coordinator_pid),'--gpu-uuid',gpu_uuid]
    restoration = str(collection/'restoration')
    restorer = common + ['restorer',*args,'--restoration-dir',restoration]
    worker = common + ['worker',*args,'--restoration-dir',restoration]
    observer = common + ['observer-supervisor',*args]
    telemetry = common + ['telemetry',*args]
    cpu,gpu = telemetry_commands(root,collection/'telemetry',python,taskset,nvidia,gpu_uuid)
    cpu.insert(cpu.index(python)+1,'-S')
    set_clock = [nvidia,'-i',gpu_uuid,'--lock-gpu-clocks=1710,1710']
    reset = [nvidia,'-i',gpu_uuid,'--reset-gpu-clocks']
    def declaration(argv, timeout, grace=0, report=None):
        return {'argv':argv,'cwd':str(root),'timeout_seconds':timeout,
                'termination_grace_seconds':grace,'output_limit_bytes':CAPTURE_BYTES,
                'report_limit_bytes':report}
    roles = {'restorer':declaration(restorer,SUPERVISOR_SECONDS,RESTORATION_GRACE_SECONDS),
             'set':declaration(set_clock,10), 'worker':declaration(worker,600),
             'reset':declaration(reset,10), 'telemetry_worker':declaration(telemetry,OBSERVER_WORKER_TIMEOUT),
             'cpu':declaration(cpu,CPU_TIMEOUT_SECONDS,report=REPORT_BYTES)}
    return {'schema_version':1,'roles':roles,'observer_argv':observer,'gpu_argv':gpu,
            'environment':env,'coordinator_pid':coordinator_pid,'gpu_uuid':gpu_uuid,
            'snapshot_sha256':snapshot_sha256,'launch_ready':False,
            'hardware_policy_verified':False,'live_role_identity_verified':False,
            'scope':'Pure fixed command and budget construction over supplied declarations; '
                    'not authenticated preparation, actual role ancestry, clock permission, '
                    'hardware restoration, quiet-window confirmation or execution.'}
