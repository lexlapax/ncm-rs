"""Linux subreaper supervision for one trusted collection worker; no GPU commands.

Run in a fresh, dedicated interpreter, not a shared application process. Surviving
supervisor handles worker death, reparented session-escaping descendants, parent
loss and interrupts. Supervisor SIGKILL/host failure are NOT contained here.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from bounded_child import run_bounded


def subreaper(value=None):
    """Narrow Linux prctl ABI: PR_GET/SET_CHILD_SUBREAPER, process-local only."""
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.prctl
    call.restype = ctypes.c_int
    call.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    current = ctypes.c_int()
    if call(37, ctypes.addressof(current), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER failed')
    if value is not None and call(36, int(value), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    return current.value


def direct_children():
    # Single-threaded caller: every direct/adopted child is listed on this task.
    raw = Path(f'/proc/self/task/{os.getpid()}/children').read_text()
    return [int(pid) for pid in raw.split()]


def cleanup_owned(seconds=2):
    """Signal only verified waitable children, never a numeric PID after reaping.

    Internal helper: caller must own every child under the fresh-supervisor
    contract. Killing adopted parents reparents their children here, including new sessions.
    Scan until empty or deadline; no claim of cleanup on timeout/ownership loss.
    """
    deadline = time.monotonic() + seconds
    signalled, reaped, errors = [], [], []
    sent = set()
    while True:
        children = direct_children()
        if not children:
            return {'complete': not errors, 'signalled': signalled, 'reaped': reaped,
                    'remaining': [], 'errors': errors}
        if time.monotonic() >= deadline:
            return {'complete': False, 'signalled': signalled, 'reaped': reaped,
                    'remaining': children, 'errors': errors + ['cleanup_deadline']}
        for pid in children:
            try:
                exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                errors.append('child_ownership_unconfirmed')
                continue  # Never signal this unowned numeric PID.
            if exited is None and pid not in sent:
                try:
                    os.kill(pid, signal.SIGKILL)
                    sent.add(pid)
                    signalled.append(pid)
                except OSError as error:
                    errors.append('signal_failed:' + type(error).__name__)
                    # Do not retry a failed signal; retain failure and wait boundedly.
                    sent.add(pid)
            # Ownership remains held: no other reaper/thread/SIGCHLD handler allowed.
            exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None:
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    reaped.append({'pid': waited, 'returncode': os.waitstatus_to_exitcode(status)})
                    sent.discard(pid)  # Later PID reuse must undergo fresh ownership validation.
        time.sleep(0.01)


def supervise(argv, *, cwd, env, output_dir, timeout_seconds, parent_pid):
    """Capture a worker then clean adopted children; any orphan makes the run fail.

    Caller supplies frozen worker argv/environment, the whole-run deadline and
    its own PID before spawning this dedicated supervisor interpreter.
    No retries, GPU policy actions or benchmark authorization are provided.
    """
    if type(parent_pid) is not int or parent_pid <= 0 or os.getppid() != parent_pid:
        raise ValueError('supervisor parent identity missing or already lost')
    if sys.platform != 'linux' or threading.current_thread() is not threading.main_thread() \
            or threading.active_count() != 1 or len(list(Path('/proc/self/task').iterdir())) != 1:
        raise ValueError('dedicated single-threaded Linux supervisor required')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL or direct_children() or subreaper() != 0:
        raise ValueError('fresh supervisor with no children/default SIGCHLD required')
    directory = Path(output_dir)
    directory.mkdir(mode=0o700)
    parent = parent_pid
    interrupted = None
    parent_lost = False
    orphan_observed = False
    saved = {}
    worker = None
    errors = []
    cleanup = {'complete': False, 'errors': ['cleanup_not_attempted']}
    restored = False

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def cancelled():
        nonlocal parent_lost, orphan_observed
        parent_lost |= os.getppid() != parent
        orphan_observed |= len(direct_children()) > 1
        return interrupted is not None or parent_lost or orphan_observed

    subreaper(1)
    try:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.signal(sig, interrupt)
            worker = run_bounded(argv, cwd=cwd, env=env,
                                 capture_path=directory / 'worker.txt',
                                 receipt_path=directory / 'worker.receipt.json',
                                 timeout_seconds=timeout_seconds, output_limit_bytes=1024 ** 2,
                                 cancel_check=cancelled)
        except BaseException as error:
            errors.append('worker_capture_error:' + type(error).__name__)
        finally:
            parent_lost |= os.getppid() != parent
            try:
                orphan_observed |= bool(direct_children())
                cleanup = cleanup_owned()
            except BaseException as error:
                errors.append('cleanup_error:' + type(error).__name__)
            if cleanup['complete']:
                try:
                    subreaper(0)
                    restored = True
                except OSError as error:
                    errors.append('subreaper_restore_error:' + type(error).__name__)
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    success = (worker is not None and worker['status'] == 'complete' and cleanup['complete']
               and restored and not errors and not parent_lost and not orphan_observed and interrupted is None)
    result = {'schema_version': 1, 'status': 'complete' if success else 'failed',
              'worker': worker, 'cleanup': cleanup, 'errors': errors,
              'parent_lost': parent_lost, 'orphan_observed': orphan_observed,
              'interrupt_signal': interrupted, 'subreaper_restored': restored,
              'launch_provenance_verified': False, 'benchmark_outputs_qualified': False,
              'scope': 'One worker and owned/adopted children while this independent supervisor lives. '
                       'No unrelated processes signalled. Supervisor SIGKILL, host failure, uninterruptible '
                       'kernel tasks and exclusive-reaper violations are not successful containment.'}
    with (directory / 'supervisor.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return result
