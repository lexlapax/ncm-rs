"""Owned set/worker/reset transaction for a dedicated Linux supervisor.

Import-only safety primitive, NOT a GPU launch entry or authorization. Callers owe
reviewed commands, policy/controller/permission checks, telemetry readiness,
enforcement/output verification and fresh availability. Tests use fake commands.
A surviving supervisor attempts reset even after partial set failure or parent loss.
Supervisor SIGKILL, host failure and uninterruptible tasks cannot be contained.
"""
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

import diagnostic_clock
from bounded_child import run_bounded
from supervised_worker import cleanup_owned, direct_children, subreaper

COMMAND_SECONDS = 10
MAX_LOCK_SECONDS = 600
CLEANUP_RESERVE_SECONDS = 10
CAPTURE_BYTES = 1024 ** 2


def command(value):
    if not isinstance(value, (list, tuple)) or not value or not all(
            isinstance(part, str) and part and '\0' not in part for part in value):
        raise ValueError('explicit nonempty argv required')
    if not Path(value[0]).is_absolute() or not os.access(value[0], os.X_OK) or not Path(value[0]).is_file():
        raise ValueError('absolute executable required')
    return list(value)


def completed(receipt):
    return (isinstance(receipt, dict) and receipt.get('status') == 'complete'
            and receipt.get('issues') == [] and type(receipt.get('returncode')) is int
            and receipt['returncode'] == 0 and receipt.get('direct_child_reaped') is True
            and receipt.get('capture_eof') is True and receipt.get('interrupt_signal') is None)


def supervise_restoration(set_argv, worker_argv, reset_argv, *, cwd, env, output_dir,
                          parent_pid, lock_seconds=MAX_LOCK_SECONDS,
                          command_seconds=COMMAND_SECONDS,
                          cleanup_reserve_seconds=CLEANUP_RESERVE_SECONDS,
                          external_cancel_check=None):
    """Attempt set once, worker at most once, reset at most twice; never retry work.

    Persist an obligation before set. All normal/exception/timeout/signal/parent-loss
    paths after arming reach reset. Only owned/adopted children are cleaned; reset
    is best effort even if that cleanup fails, with failure retained. Set/worker
    cancellation reserves time before the absolute reset-request deadline. Missing
    or late first-reset spawn evidence fails the transaction but never suppresses
    reset. Scheduling/IO/cleanup may overrun: no hard real-time guarantee. Reset
    completion has its own bounded timeout. No frozen-file hashing occurs.
    Successful reset exit is NOT proof of enforcement or restored hardware policy.
    Optional external_cancel_check is a trusted nonblocking Boolean predicate for
    observer failure. True/errors latch cancellation; reset never calls it. This
    hook does not itself establish telemetry readiness, identity or full coverage.
    """
    if sys.platform != 'linux' or not sys.flags.no_site or threading.current_thread() is not threading.main_thread() \
            or threading.active_count() != 1 or len(list(Path('/proc/self/task').iterdir())) != 1:
        raise ValueError('dedicated single-threaded Linux -S interpreter required')
    if type(parent_pid) is not int or parent_pid <= 0 or os.getppid() != parent_pid:
        raise ValueError('explicit live parent required')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL or direct_children() or subreaper() != 0:
        raise ValueError('exclusive fresh reaper required')
    set_argv, worker_argv, reset_argv = map(command, (set_argv, worker_argv, reset_argv))
    for value, maximum in ((lock_seconds, MAX_LOCK_SECONDS), (command_seconds, COMMAND_SECONDS)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('invalid bounded transaction timeout')
    if (type(cleanup_reserve_seconds) not in (int, float)
            or not math.isfinite(cleanup_reserve_seconds)
            or not 0 < cleanup_reserve_seconds < lock_seconds):
        raise ValueError('invalid cleanup reserve')
    if external_cancel_check is not None and not callable(external_cancel_check):
        raise ValueError('external_cancel_check must be callable')
    if not isinstance(env, dict) or not all(isinstance(k, str) and k and '=' not in k and '\0' not in k
            and isinstance(v, str) and '\0' not in v for k, v in env.items()):
        raise ValueError('explicit valid environment required')
    env = dict(env)
    cwd = Path(cwd).resolve(strict=True)
    directory = Path(output_dir).absolute()
    if not cwd.is_dir() or os.path.lexists(directory) or not directory.parent.is_dir():
        raise ValueError('valid cwd and fresh output directory required')
    directory = directory.resolve()
    directory.mkdir(mode=0o700)
    lifetime = diagnostic_clock.begin()
    saved, errors, cleanups, resets = {}, [], [], []
    interrupted = None
    parent_lost = orphan_observed = armed = role_acquired = restored = False
    set_receipt = worker_receipt = None
    deadline = reset_deadline = None
    reset_started = None
    reset_request_within_deadline = False
    external_cancelled = False

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def cancelled():
        nonlocal parent_lost, orphan_observed, external_cancelled
        parent_lost |= os.getppid() != parent_pid
        orphan_observed |= len(direct_children()) > 1
        if external_cancel_check is not None and not external_cancelled:
            try:
                status = external_cancel_check()
                if type(status) is not bool:
                    raise ValueError('external cancellation status must be Boolean')
                external_cancelled = status
            except BaseException as error:
                external_cancelled = True
                errors.append('external_cancel_error:' + type(error).__name__)
                raise
        return interrupted is not None or parent_lost or orphan_observed or external_cancelled

    def clean(label):
        nonlocal orphan_observed
        try:
            orphan_observed |= bool(direct_children())
            result = cleanup_owned()
        except BaseException as error:
            result = {'complete': False, 'errors': [type(error).__name__]}
        cleanups.append({'after': label, 'result': result})
        if not result['complete']:
            errors.append('owned_cleanup_failed:' + label)
        return result['complete']

    def capture(argv, label, timeout, *, restoring=False):
        nonlocal interrupted
        receipt = run_bounded(argv, cwd=cwd, env=env,
                              capture_path=directory / (label + '.txt'),
                              receipt_path=directory / (label + '.receipt.json'),
                              timeout_seconds=timeout, output_limit_bytes=CAPTURE_BYTES,
                              sequence_deadline=None if restoring else deadline,
                              cancel_check=None if restoring else cancelled)
        if receipt['interrupt_signal'] is not None:
            interrupted = receipt['interrupt_signal']
        return receipt

    try:
        subreaper(1)
        role_acquired = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            saved[sig] = signal.signal(sig, interrupt)
        if not cancelled():
            # Failure to write this marker prevents set. No live-policy action
            # precedes a successfully closed marker; no environment values stored.
            reset_deadline = time.monotonic() + lock_seconds
            deadline = reset_deadline - cleanup_reserve_seconds
            with (directory / 'obligation.json').open('x') as file:
                json.dump({'schema_version': 2, 'armed_before_set': True,
                           'set_argv': set_argv, 'worker_argv': worker_argv, 'reset_argv': reset_argv,
                           'lock_seconds': lock_seconds, 'command_seconds': command_seconds,
                           'cleanup_reserve_seconds': cleanup_reserve_seconds,
                           'work_deadline_monotonic_seconds': deadline,
                           'reset_deadline_monotonic_seconds': reset_deadline,
                           'maximum_reset_attempts': 2, 'clock': diagnostic_clock.stamp()}, file, indent=2)
                file.write('\n')
            armed = True
            set_receipt = capture(set_argv, 'set', command_seconds)
            set_clean = clean('set')
            if completed(set_receipt) and set_clean and not cancelled() and time.monotonic() < deadline:
                worker_receipt = capture(worker_argv, 'worker', lock_seconds)
                # Catch observer loss during final worker cleanup/receipt writing.
                # A complete worker receipt cannot erase failed collection state.
                cancelled()
    except BaseException as error:
        errors.append('transaction_error:' + type(error).__name__)
    finally:
        if role_acquired:
            clean('before_reset')
        if armed:
            # Evidence/clock failures must not bypass the armed cleanup action.
            try:
                reset_started = diagnostic_clock.stamp()
                if reset_deadline is None or time.monotonic() >= reset_deadline:
                    errors.append('reset_deadline_reached_or_unavailable')
            except BaseException as error:
                errors.append('reset_clock_error:' + type(error).__name__)
            for attempt in range(1, 3):
                receipt = None
                try:
                    # Never suppress restoration because parent/worker is gone or
                    # an earlier interrupt is pending. A new interrupt is retained
                    # by bounded capture; the one cleanup-only retry still applies.
                    receipt = capture(reset_argv, f'reset-{attempt}', command_seconds, restoring=True)
                except BaseException as error:
                    errors.append('reset_capture_error:' + type(error).__name__)
                resets.append(receipt)
                if attempt == 1:
                    # The pre-capture reset stamp can precede slow file opens or
                    # spawn. Only the later owned Popen return bounds submission;
                    # it still does not prove execution or hardware restoration.
                    spawn = receipt.get('spawn_returned_monotonic_seconds') if isinstance(receipt, dict) else None
                    pid = receipt.get('pid') if isinstance(receipt, dict) else None
                    reset_request_within_deadline = (
                        type(pid) is int and pid > 0 and type(spawn) in (int, float)
                        and math.isfinite(spawn) and reset_deadline is not None
                        and reset_deadline - lock_seconds <= spawn <= reset_deadline)
                    if not reset_request_within_deadline:
                        errors.append('reset_request_deadline_unverified_or_missed')
                reset_clean = clean(f'reset-{attempt}')
                if completed(receipt) and reset_clean:
                    break
                errors.append(f'reset_attempt_failed:{attempt}')
            if not resets or not completed(resets[-1]) or not reset_clean:
                try:
                    print('RESET UNCONFIRMED OR OWNED CLEANUP FAILED: manual inspection required; no more workloads.',
                          file=sys.stderr, flush=True)
                except BaseException as error:
                    # Parent loss can close stderr. Still restore the process role
                    # and persist failed reset evidence; never turn it into success.
                    errors.append('warning_delivery_error:' + type(error).__name__)
        parent_lost |= os.getppid() != parent_pid
        if role_acquired:
            try:
                if not direct_children() and all(c['result']['complete'] for c in cleanups):
                    subreaper(0)
                    restored = True
            except BaseException as error:
                errors.append('subreaper_restore_error:' + type(error).__name__)
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    reset_ok = bool(resets) and completed(resets[-1])
    success = (armed and completed(set_receipt) and completed(worker_receipt) and reset_ok
               and restored and not errors and not parent_lost and not orphan_observed
               and not external_cancelled and interrupted is None)
    result = {'schema_version': 2, 'status': 'complete' if success else 'failed',
              'armed': armed, 'set': set_receipt, 'worker': worker_receipt, 'reset_attempts': resets,
              'reset_command_succeeded': reset_ok, 'reset_started_clock': reset_started,
              'work_deadline_monotonic_seconds': deadline,
              'reset_deadline_monotonic_seconds': reset_deadline,
              'cleanup_reserve_seconds': cleanup_reserve_seconds,
              'reset_request_within_deadline': reset_request_within_deadline,
              'cleanups': cleanups, 'errors': errors,
              'parent_lost': parent_lost, 'orphan_observed': orphan_observed,
              'external_cancel_check_enabled': external_cancel_check is not None,
              'external_cancelled': external_cancelled,
              'interrupt_signal': interrupted, 'subreaper_restored': restored,
              'hardware_policy_restoration_verified': False, 'clock_enforcement_verified': False,
              'launch_provenance_verified': False, 'benchmark_outputs_qualified': False,
              'lifetime': diagnostic_clock.finish(lifetime),
              'scope': 'Owned command transaction only, not launch authorization or hardware verification. '
                       'Early work cancellation reserves cleanup time; late/unverified reset submission fails '
                       'without suppressing restoration. No hard real-time reset guarantee. Surviving supervisor required.'}
    with (directory / 'restoration.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    return result
