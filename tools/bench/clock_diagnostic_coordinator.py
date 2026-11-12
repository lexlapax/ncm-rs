"""Readiness-gated clock-restorer lifecycle; import-only, no policy commands.

Caller owes frozen roles/identities, policy/permission checks and availability.
Do not place this coordinator under a generic adoption-and-SIGKILL supervisor:
the independent restorer must survive coordinator loss long enough to reset.
The observer argv must run a separate supervisor owning the telemetry worker;
only that branch may use generic adopted-child cleanup. Caller binds its roles.
No ancestry safety, full telemetry qualification or hardware restoration proof
is supplied here; those remain launch-entry and completion requirements.
"""
from pathlib import Path
import sys

from clock_restoration_capture import capture_restorer
from readonly_trace_coordinator import _coordinate_action, validate_ready
from readonly_trace_preflight import require
from supervised_worker import subreaper


def validate_supervised_ready(record, supervisor_pid, source, uuid):
    """Bind readiness to the sole live worker of the owned observer supervisor.

    The coordinator holds the supervisor waitable before/after this check. These
    procfs snapshots establish a sampled relationship, not immutable identity or
    continuous liveness. Frozen entry roles and postflight receipts are still owed.
    No signal is sent to this indirectly owned worker by the coordinator.
    """
    pid = record['worker_pid']
    require(type(pid) is int and pid > 0 and pid != supervisor_pid,
            'readiness needs a distinct supervised telemetry worker')
    validate_ready(record, pid, source, uuid)
    children = Path(f'/proc/{supervisor_pid}/task/{supervisor_pid}/children').read_text().split()
    require(children == [str(pid)], 'telemetry worker is not the sole observer-supervisor child')
    stat = Path(f'/proc/{pid}/stat').read_text()
    fields = stat.rsplit(')', 1)[-1].split()
    require(stat.split(' ', 1)[0] == str(pid) and len(fields) >= 2
            and fields[0] not in ('Z', 'X', 'x') and fields[1] == str(supervisor_pid),
            'telemetry worker exited or changed parent before readiness')


def coordinate(restorer_argv, observer_argv, *, cwd, env, output_dir,
               expected_cpu_source, gpu_uuid):
    """Start one restorer only after telemetry readiness; never retry the action.

    observer_argv must name a dedicated independent subreaper wrapping telemetry,
    exiting nonzero on any supervisor failure. It must not own the restorer.
    Ordinary failed restorer capture preserves the sampler lifetime. Observer
    failure or interruption cancels the restorer through its cleanup-aware outer
    capture. Reject an existing local subreaper role before either branch starts;
    this does not inspect ancestors or prove restoration-aware launch ancestry.
    A successful lifecycle result does not qualify clocks or codec output.
    """
    require(sys.platform == 'linux', 'Linux clock coordinator required')
    require(subreaper() == 0, 'clock coordinator must not already be a child subreaper')
    directory = Path(output_dir).absolute()
    require(isinstance(restorer_argv, (list, tuple)) and restorer_argv
            and all(isinstance(arg, str) and arg and '\0' not in arg for arg in restorer_argv)
            and Path(restorer_argv[0]).is_absolute()
            and str(directory / 'restoration') in restorer_argv,
            'restorer must have an absolute executable and exclusive restoration directory argument')
    restorer_argv = list(restorer_argv)

    def action(monitor):
        receipt = capture_restorer(restorer_argv, cwd=cwd, env=dict(env),
                                   capture_path=directory / 'restorer.capture',
                                   receipt_path=directory / 'restorer.receipt.json',
                                   cancel_check=monitor)
        return receipt, receipt['interrupt_signal']

    return _coordinate_action(observer_argv, cwd=cwd, env=env, output_dir=directory,
                              expected_cpu_source=expected_cpu_source, gpu_uuid=gpu_uuid,
                              run_action=action, status_key='restoration_supervisor_status',
                              ready_validator=validate_supervised_ready)
