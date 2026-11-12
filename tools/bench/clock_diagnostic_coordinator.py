"""Readiness-gated clock-restorer lifecycle; import-only, no policy commands.

Caller owes frozen roles/identities, policy/permission checks and availability.
Do not place this coordinator under a generic adoption-and-SIGKILL supervisor:
the independent restorer must survive coordinator loss long enough to reset.
Observer orphan/escape cleanup still needs a separately supervised observer branch.
No ancestry safety, full telemetry qualification or hardware restoration proof
is supplied here; those remain launch-entry and completion requirements.
"""
from pathlib import Path

from clock_restoration_capture import capture_restorer
from readonly_trace_coordinator import _coordinate_action
from readonly_trace_preflight import require


def coordinate(restorer_argv, observer_argv, *, cwd, env, output_dir,
               expected_cpu_source, gpu_uuid):
    """Start one restorer only after telemetry readiness; never retry the action.

    Ordinary failed restorer capture preserves the sampler lifetime. Observer
    failure or interruption cancels the restorer through its cleanup-aware outer
    capture. A successful lifecycle result does not qualify clocks or codec output.
    """
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
                              run_action=action, status_key='restoration_supervisor_status')
