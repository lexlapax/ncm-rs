"""Import-only outer capture budget for a trusted clock-restoration supervisor.

No clock commands or launch authorization. Caller supplies frozen roles/identities,
explicit environments, readiness and cancellation, then validates restoration and
full collection evidence. Default native/read-only capture budgets are unchanged.
"""
import sys

from bounded_child import run_bounded

SUPERVISOR_SECONDS = 660
RESTORATION_GRACE_SECONDS = 60
CAPTURE_BYTES = 1024 ** 2
WARNING = ('RESTORATION SUPERVISION FAILED/CANCELLED: verify retained cleanup evidence '
           'and hardware policy; no more workloads.')


def capture_restorer(argv, *, cwd, env, capture_path, receipt_path, cancel_check):
    """Own one supervisor, allow TERM cleanup, then bounded escalation if needed.

    The 660-second outer budget is not an extension of the inner 600-second reset
    request deadline. Cancellation grants up to 60 additional seconds for cleanup;
    no real-time or restoration guarantee is made if this allowance is exhausted.
    The raw receipt proves neither correct worker roles nor restored GPU policy.
    cancel_check must be a trusted, nonblocking Boolean predicate.
    """
    if not callable(cancel_check):
        raise ValueError('mandatory cancellation monitor required')
    def cancellation():
        status = cancel_check()
        if type(status) is not bool:
            raise ValueError('cancellation status must be Boolean')
        return status
    try:
        receipt = run_bounded(argv, cwd=cwd, env=env, capture_path=capture_path,
                              receipt_path=receipt_path, timeout_seconds=SUPERVISOR_SECONDS,
                              output_limit_bytes=CAPTURE_BYTES, cancel_check=cancellation,
                              termination_grace_seconds=RESTORATION_GRACE_SECONDS)
    except BaseException:
        print(WARNING, file=sys.stderr, flush=True)
        raise
    if receipt['status'] != 'complete':
        print(WARNING, file=sys.stderr, flush=True)
    return receipt
