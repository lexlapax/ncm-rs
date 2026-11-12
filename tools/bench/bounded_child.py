"""Linux-only bounded subprocess capture primitive, not a diagnostic launcher.

No GPU commands or policy changes are built in. Caller owns authorization, frozen
identities, affinity, sequence quotas and workload qualification. SIGKILL/host
failure and descendants that escape the created session are not contained here.
"""
import datetime
import hashlib
import diagnostic_clock
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fingerprint_report(path, limit_bytes):
    """Bounded post-exit hash of a regular, non-symlink report in a trusted directory.

    No live disk quota or independent proof of the file's producer is implied.
    Refuse FIFOs/devices and visible mutation while reading. Caller must prevent
    concurrent external writers; metadata checks are not an adversarial snapshot.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit_bytes:
            raise ValueError('report must be a nonempty bounded regular file')
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(fd, min(65536, limit_bytes + 1 - size)):
            size += len(chunk)
            if size > limit_bytes:
                raise ValueError('report exceeds byte limit')
            digest.update(chunk)
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        current = os.stat(path, follow_symlinks=False)
        if size != before.st_size or any(getattr(before, k) != getattr(s, k)
                                        for s in (after, current) for k in fields):
            raise ValueError('report changed while hashing')
        return {'path': str(path), 'size_bytes': size, 'sha256': digest.hexdigest(),
                'device': before.st_dev, 'inode': before.st_ino,
                'post_exit_hash_verified': True, 'producer_identity_verified': False}
    finally:
        os.close(fd)


def run_bounded(argv, *, cwd, env, capture_path, receipt_path, timeout_seconds,
                output_limit_bytes, sequence_deadline=None, report_path=None,
                report_limit_bytes=16 * 1024 ** 2, cancel_check=None, observe_child=None,
                termination_grace_seconds=0):
    """Capture one owned session; return a failure receipt rather than retry.

    SIGINT/SIGTERM temporarily belong to this main-thread collector and result in
    an interrupted receipt. The caller MUST stop its sequence for any non-complete
    status. Only the direct child is waitable/reaped; no escaped-descendant or
    workload-completion guarantee is made. Existing paths are never overwritten.
    Optional report_path must be an absent absolute argv argument beside capture;
    its post-exit hash is bound into the receipt, not proof of producer/quality.
    Its size limit bounds post-exit reading, NOT the child's live disk writes.
    cancel_check is an optional trusted, nonblocking supervisor predicate, checked
    before spawn and while collecting; true means failure and owned-child cleanup.
    observe_child is an optional trusted bounded callback receiving the owned,
    unreaped live PID during capture; exceptions fail capture and clean the child.
    Optional termination_grace_seconds (0..10) gives a live supervisor SIGTERM
    before final group cleanup. Default/native jobs still use immediate cleanup.
    Grace is additional cleanup time, not an extension of the capture deadline.
    """
    if type(termination_grace_seconds) not in (int, float) or not math.isfinite(termination_grace_seconds) \
            or not 0 <= termination_grace_seconds <= 10:
        raise ValueError('invalid termination grace')
    if observe_child is not None and not callable(observe_child):
        raise ValueError('observe_child must be callable')
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError('cancel_check must be callable')
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
    if report_path is not None:
        if type(report_limit_bytes) is not int or report_limit_bytes <= 0:
            raise ValueError('invalid report limit')
        report_path = Path(report_path)
        if not report_path.is_absolute() or str(report_path) not in argv:
            raise ValueError('report must be an explicit absolute argv argument')
        if report_path.parent.resolve(strict=True) != capture_path.parent.resolve(strict=True):
            raise ValueError('report must be beside the capture in a trusted directory')
        if report_path.resolve() in (capture_path.resolve(), receipt_path.resolve()):
            raise ValueError('report and collector outputs must be separate files')
        if os.path.lexists(report_path):
            raise ValueError('report path already exists')

    child = None
    interrupted = None
    saved_handlers = {}
    issues = []
    captured = observed = 0
    capture_digest = hashlib.sha256()
    eof = False
    child_reaped = False
    group_kill = 'not_attempted'
    termination_signal = 'not_attempted'
    termination_grace_exhausted = False
    returncode = None
    lifetime = diagnostic_clock.begin()
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
                    # Output/handler setup can consume the remaining allowance or
                    # observe cancellation. Recheck before attempting any spawn.
                    # Events arriving during Popen still use the cleanup path.
                    if interrupted is not None:
                        issue('interrupted')
                    elif cancel_check is not None and cancel_check():
                        issue('external_cancelled')
                    elif time.monotonic() >= deadline:
                        issue(deadline_reason)
                    else:
                        child = subprocess.Popen(list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        os.set_blocking(child.stdout.fileno(), False)
                        selector.register(child.stdout, selectors.EVENT_READ)
                        while not issues:
                            if interrupted is not None:
                                issue('interrupted')
                                break
                            if cancel_check is not None and cancel_check():
                                issue('external_cancelled')
                                break
                            if time.monotonic() >= deadline:
                                issue(deadline_reason)
                                break
                            # Do NOT poll()/wait() here: preserve the leader PID until
                            # the final group signal, preventing a PID-reuse kill race.
                            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            if exited is not None:
                                break
                            if observe_child is not None:
                                observe_child(child.pid)
                            for _key, _mask in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                                read_chunk()
                except BaseException as error:
                    issue('capture_error:' + type(error).__name__)
                finally:
                    if child is not None:
                        owned = True
                        try:
                            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            if exited is None and termination_grace_seconds:
                                try:
                                    os.killpg(child.pid, signal.SIGTERM)
                                    termination_signal = 'sent'
                                except ProcessLookupError:
                                    termination_signal = 'no_group'
                                except OSError as error:
                                    termination_signal = type(error).__name__
                                    issue('graceful_signal_failed')
                                grace_end = time.monotonic() + termination_grace_seconds
                                while exited is None and time.monotonic() < grace_end:
                                    time.sleep(.02)
                                    exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                                termination_grace_exhausted = exited is None
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
                    report_capture = None
                    if report_path is not None:
                        report_capture = {'path': str(report_path), 'post_exit_hash_verified': False,
                                          'producer_identity_verified': False}
                        if not issues and child_reaped and eof:
                            try:
                                report_capture = fingerprint_report(report_path, report_limit_bytes)
                            except (OSError, ValueError) as error:
                                issue('report_error:' + type(error).__name__)
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
                        'termination_grace_seconds': termination_grace_seconds,
                        'termination_signal': termination_signal,
                        'termination_grace_exhausted': termination_grace_exhausted,
                        'interrupt_signal': interrupted, 'workload_output_qualified': False,
                        'executable_identity_verified': False,
                        'process_observer_enabled': observe_child is not None,
                        'report_capture': report_capture,
                        'report_limit_bytes': report_limit_bytes if report_path is not None else None,
                        'scope': 'Direct-child exit and bounded merged stdout/stderr only; not workload '
                                 'qualification, launch identity proof, or escaped-descendant containment.',
                    }
                    result['lifetime'] = diagnostic_clock.finish(lifetime)
                    json.dump(result, receipt, indent=2)
                    receipt.write('\n')
                    receipt.flush()
            finally:
                restore_handlers()
    return result
