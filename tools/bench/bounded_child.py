"""Linux-only bounded subprocess capture primitive, not a diagnostic launcher.

No GPU commands or policy changes are built in. Caller owns authorization, frozen
identities, affinity, sequence quotas and workload qualification. SIGKILL/host
failure and descendants that escape the created session are not contained here.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_bounded(argv, *, cwd, env, capture_path, receipt_path, timeout_seconds,
                output_limit_bytes, sequence_deadline=None):
    """Capture one owned session; return a failure receipt rather than retry.

    SIGINT/SIGTERM temporarily belong to this main-thread collector and result in
    an interrupted receipt. The caller MUST stop its sequence for any non-complete
    status. Only the direct child is waitable/reaped; no escaped-descendant or
    workload-completion guarantee is made. Existing paths are never overwritten.
    """
    if sys.platform != 'linux' or not hasattr(os, 'WNOWAIT'):
        raise ValueError('Linux waitid(WNOWAIT) is required')
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ValueError('collector requires a single-threaded main process')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise ValueError('collector requires default SIGCHLD and exclusive child-reaping ownership')
    if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise ValueError('expected a nonempty argv list, never a shell command')
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError('explicit child environment required')
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('invalid child timeout')
    if type(output_limit_bytes) is not int or output_limit_bytes <= 0:
        raise ValueError('invalid output limit')
    if sequence_deadline is not None and (type(sequence_deadline) not in (int, float)
            or not math.isfinite(sequence_deadline) or sequence_deadline <= time.monotonic()):
        raise ValueError('invalid or expired sequence deadline')
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError('child cwd must be a directory')
    capture_path, receipt_path = Path(capture_path), Path(receipt_path)
    if capture_path.resolve() == receipt_path.resolve():
        raise ValueError('capture and receipt must be separate files')

    child = None
    interrupted = None
    saved_handlers = {}
    issues = []
    captured = observed = 0
    capture_digest = hashlib.sha256()
    eof = False
    child_reaped = False
    group_kill = 'not_attempted'
    returncode = None
    started = time.monotonic()
    child_deadline = started + timeout_seconds
    deadline = min(child_deadline, sequence_deadline) if sequence_deadline is not None else child_deadline
    deadline_reason = 'sequence_deadline' if deadline < child_deadline else 'child_timeout'
    started_utc = utc_now()

    def issue(reason):
        if reason not in issues:
            issues.append(reason)

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def restore_handlers():
        for sig in tuple(saved_handlers):
            signal.signal(sig, saved_handlers[sig])
            del saved_handlers[sig]

    # Exclusive creation happens before signal ownership or child creation. If the
    # second open fails, the first empty file remains as incomplete local evidence.
    with capture_path.open('xb', buffering=0) as output, receipt_path.open('x') as receipt:
        with selectors.DefaultSelector() as selector:
            def read_chunk():
                nonlocal captured, observed, eof
                try:
                    data = os.read(child.stdout.fileno(), 65536)
                except BlockingIOError:
                    return
                if not data:
                    eof = True
                    selector.unregister(child.stdout)
                    return
                observed += len(data)
                keep = data[:max(0, output_limit_bytes - captured)]
                while keep:
                    written = output.write(keep)
                    if not written:
                        raise OSError('capture write made no progress')
                    capture_digest.update(keep[:written])
                    captured += written
                    keep = keep[written:]
                if observed > output_limit_bytes:
                    issue('output_limit')

            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    saved_handlers[sig] = signal.signal(sig, interrupt)
                try:
                    child = subprocess.Popen(list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    os.set_blocking(child.stdout.fileno(), False)
                    selector.register(child.stdout, selectors.EVENT_READ)
                    while not issues:
                        if interrupted is not None:
                            issue('interrupted')
                            break
                        if time.monotonic() >= deadline:
                            issue(deadline_reason)
                            break
                        # Do NOT poll()/wait() here: preserve the leader PID until
                        # the final group signal, preventing a PID-reuse kill race.
                        exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                        if exited is not None:
                            break
                        for _key, _mask in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                            read_chunk()
                except BaseException as error:
                    issue('capture_error:' + type(error).__name__)
                finally:
                    if child is not None:
                        owned = True
                        try:
                            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                        except OSError:
                            # Never use a numeric group ID if another reaper has
                            # invalidated ownership. This is an unsupported caller.
                            owned = False
                            group_kill = 'not_sent_ownership_unconfirmed'
                            issue('child_ownership_unconfirmed')
                            # No signal is permitted after ownership loss. Let
                            # Popen settle its bookkeeping without interpreting
                            # ECHILD's synthesized status as a verified exit.
                            child.poll()
                        if owned:
                            try:
                                # The leader remains waitable until this final
                                # signal. Never killpg after wait(): its PID may
                                # then be reused by an unrelated process group.
                                os.killpg(child.pid, signal.SIGKILL)
                                group_kill = 'sent'
                            except ProcessLookupError:
                                group_kill = 'no_group'
                            except OSError as error:
                                group_kill = type(error).__name__
                                issue('group_cleanup_failed')
                            try:
                                returncode = child.wait(timeout=2)
                                child_reaped = True
                            except subprocess.TimeoutExpired:
                                issue('direct_child_not_reaped')
                        # Drain already-buffered output after termination, still
                        # enforcing the cap. An escaped pipe holder cannot hang us.
                        drain_deadline = time.monotonic() + 2
                        try:
                            while not eof and not any(i.startswith('capture_error:') or i == 'output_limit' for i in issues):
                                if time.monotonic() >= drain_deadline:
                                    issue('capture_pipe_not_closed')
                                    break
                                for _key, _mask in selector.select(0.05):
                                    read_chunk()
                        except BaseException as error:
                            issue('capture_error:' + type(error).__name__)
                        finally:
                            child.stdout.close()
                    # Restore before serializing the receipt: a late interrupt
                    # must not be swallowed after the status was computed.
                    restore_handlers()
                    if interrupted is not None:
                        issue('interrupted')
                    if returncode not in (None, 0):
                        issue('nonzero_exit')
                    result = {
                        'schema_version': 1, 'status': 'complete' if not issues and child_reaped and eof else 'failed',
                        'issues': issues, 'argv': list(argv), 'cwd': str(cwd),
                        'pid': child.pid if child is not None else None, 'returncode': returncode,
                        'started_utc': started_utc, 'finished_utc': utc_now(),
                        'elapsed_seconds_including_cleanup': time.monotonic() - started,
                        'timeout_seconds': timeout_seconds, 'output_limit_bytes': output_limit_bytes,
                        'captured_bytes': captured, 'observed_bytes': observed,
                        'capture_sha256': capture_digest.hexdigest(), 'capture_eof': eof,
                        'group_kill': group_kill, 'direct_child_reaped': child_reaped,
                        'interrupt_signal': interrupted, 'workload_output_qualified': False,
                        'executable_identity_verified': False,
                        'scope': 'Direct-child exit and bounded merged stdout/stderr only; not workload '
                                 'qualification, launch identity proof, or escaped-descendant containment.',
                    }
                    json.dump(result, receipt, indent=2)
                    receipt.write('\n')
                    receipt.flush()
            finally:
                restore_handlers()
    return result
