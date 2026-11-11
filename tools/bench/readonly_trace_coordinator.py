"""Compose fixed sequence and observer lifetimes inside a separately supervised worker.

Import-only: caller still owes the frozen launch contract, utility/process/library
provenance, availability confirmation and a dedicated surviving subreaper. This
module does not authorize collection or provide complete telemetry qualification.
"""
import csv
from datetime import datetime
import io
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from bounded_child import fingerprint_report, utc_now
from readonly_trace_preflight import require
from readonly_trace_sequence import run_sequence, validate_jobs, validate_report_paths
from readonly_trace_telemetry import GPU_HEADER, validate_cpu_prefix, validate_gpu_prefix
from run_phase2_linux_comparison import observer_affinity
from supervised_worker import direct_children

OBSERVER_CPU = 0
STARTUP_SECONDS = 20  # includes interpreter startup; observer's own allowance stays 15
OBSERVER_SECONDS = 950  # 900-second sampling plus startup headroom; cleanup is additional
OUTPUT_BYTES = 1024 ** 2
JSON_BYTES = 1024 ** 2


def read_record(path):
    """Trusted exclusive evidence directory; reject oversized/nonregular leaves."""
    identity = fingerprint_report(path, JSON_BYTES)
    with Path(path).open('rb') as file:
        raw = file.read(JSON_BYTES + 1)
    require(len(raw) == identity['size_bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256'],
            'record changed while reading')
    return json.loads(raw)


def validate_ready(record, pid, source, uuid):
    require(type(record['schema_version']) is int and record['schema_version'] == 1
            and record['kind'] == 'telemetry_ready'
            and type(record['worker_pid']) is int and record['worker_pid'] == pid
            and record['observer_affinity'] == [OBSERVER_CPU]
            and type(record['gpu_pid']) is int and record['gpu_pid'] > 0 and record['gpu_pid'] != pid,
            'observer readiness identity/affinity mismatch')
    validate_cpu_prefix([json.dumps(record['cpu_metadata']),
                         json.dumps({'kind': 'sample', 'index': 0})], source)
    rows = io.StringIO()
    csv.writer(rows).writerows([GPU_HEADER, record['first_gpu_row']])
    validate_gpu_prefix(rows.getvalue().splitlines(), uuid)
    require(datetime.fromisoformat(record['local_time_with_offset']).utcoffset() is not None,
            'observer readiness needs an explicit timezone')


def finish_observer(child, abort):
    """Keep leader waitable until final group cleanup; never signal after reaping.

    On cancellation allow the observer to clean its two children first. If it
    cannot, the independent subreaper must clean adopted descendants and fail.
    """
    try:
        exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        child.poll()  # bookkeeping only, no synthetic exit status accepted
        return {'verified': False, 'reason': 'ownership_unconfirmed'}
    if abort and exited is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = time.monotonic() + 6
        while exited is None and time.monotonic() < end:
            time.sleep(.02)
            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    # Final group signal BEFORE reaping also removes surviving group members.
    escalated = exited is None
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    code = child.wait(timeout=2)
    return {'verified': True, 'returncode': code, 'kill_escalation': escalated}


def coordinate(jobs, observer_argv, *, cwd, env, output_dir, expected_cpu_source, gpu_uuid,
               process_observer_factory=None):
    """Start jobs only after readiness; wait for observation after ordinary job failure.

    An observer error/exit/deadline, cancellation or coordinator exception stops
    submissions. There are no retries. Interrupts caught by run_bounded inside a
    job are propagated to the coordinator instead of accidentally waiting 900s.
    All output is private partial evidence until separate qualification/audit.
    """
    require(sys.platform == 'linux' and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1 and len(list(Path('/proc/self/task').iterdir())) == 1
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL and not direct_children(),
            'fresh single-threaded Linux coordinator with exclusive reaping required')
    require(process_observer_factory is None or callable(process_observer_factory),
            'process_observer_factory must be callable')
    jobs = validate_jobs(jobs)
    directory = Path(output_dir).absolute()
    sequence_dir, telemetry_dir = directory / 'sequence', directory / 'telemetry'
    validate_report_paths(jobs, sequence_dir)
    require(isinstance(observer_argv, (list, tuple)) and observer_argv
            and all(isinstance(arg, str) and arg and '\0' not in arg for arg in observer_argv)
            and str(telemetry_dir) in observer_argv, 'observer must have its exclusive directory argument')
    require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
            and not any(k.startswith(('LD_', 'CUDA_', 'CUPTI_', 'NVTX_', 'NCM_', 'CUBLAS_')) and v
                        for k, v in env.items()), 'explicit observer environment without runtime controls required')
    cwd = Path(cwd).resolve(strict=True)
    require(cwd.is_dir(), 'coordinator cwd must be a directory')
    directory.mkdir(mode=0o700)
    started, started_utc = time.monotonic(), utc_now()
    interrupted = None
    saved = {}
    errors = []
    observer = None
    shutdown = None
    ready = None
    sequence = None
    telemetry = None
    finished = False
    capture = directory / 'observer-worker.txt'

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def issue(reason):
        if reason not in errors:
            errors.append(reason)

    def monitor():
        nonlocal finished
        if interrupted is not None:
            return True
        if time.monotonic() - started >= OBSERVER_SECONDS:
            issue('observer_deadline')
            return True
        if capture.stat().st_size > OUTPUT_BYTES:
            issue('observer_worker_output_limit')
            return True
        if os.waitid(os.P_PID, observer.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            finished = True
            return True
        return False

    with observer_affinity(OBSERVER_CPU), capture.open('xb', buffering=0) as output:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.signal(sig, interrupt)
            if interrupted is None and time.monotonic() - started < STARTUP_SECONDS:
                observer = subprocess.Popen(observer_argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                while not monitor():
                    if time.monotonic() - started >= STARTUP_SECONDS:
                        issue('readiness_deadline')
                        break
                    path = telemetry_dir / 'ready.json'
                    if os.path.lexists(path):
                        ready = read_record(path)
                        validate_ready(ready, observer.pid, expected_cpu_source, gpu_uuid)
                        # Validation must not bypass a deadline, cancellation or worker exit.
                        if monitor() or time.monotonic() - started >= STARTUP_SECONDS:
                            issue('observer_not_ready_at_launch')
                            break
                        sequence = run_sequence(jobs, sequence_dir, cancel_check=monitor,
                                                process_observer_factory=process_observer_factory)
                        for job in sequence['jobs']:
                            if job['receipt']['interrupt_signal'] is not None:
                                interrupted = job['receipt']['interrupt_signal']
                        # A benchmark error is not permission to truncate the sampler.
                        while not monitor():
                            time.sleep(.05)
                        break
                    time.sleep(.02)
            else:
                issue('cancelled_or_expired_before_observer_spawn')
        except BaseException as error:
            issue('coordinator_error:' + type(error).__name__)
        finally:
            if observer is not None:
                try:
                    shutdown = finish_observer(observer, abort=not finished)
                except BaseException as error:
                    issue('observer_cleanup_error:' + type(error).__name__)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    if shutdown is not None and shutdown['verified']:
        try:
            telemetry = read_record(telemetry_dir / 'telemetry.json')
            require(telemetry['status'] == 'complete' and telemetry['ready'] is True,
                    'observer lifecycle did not complete')
        except (OSError, ValueError, KeyError, TypeError) as error:
            issue('telemetry_result_error:' + type(error).__name__)
    if capture.stat().st_size > OUTPUT_BYTES:
        issue('observer_worker_output_limit')
    success = (sequence is not None and sequence['status'] == 'complete' and ready is not None
               and finished and shutdown is not None and shutdown['verified']
               and shutdown['returncode'] == 0 and not shutdown['kill_escalation']
               and telemetry is not None and not errors and interrupted is None)
    result = {'schema_version': 1, 'status': 'complete' if success else 'failed',
              'started_utc': started_utc, 'finished_utc': utc_now(),
              'elapsed_seconds_including_cleanup': time.monotonic() - started,
              'observer_pid': observer.pid if observer else None, 'observer_shutdown': shutdown,
              'sequence_status': sequence['status'] if sequence is not None else None,
              'telemetry_status': telemetry.get('status') if isinstance(telemetry, dict) else None,
              'interrupt_signal': interrupted, 'errors': errors,
              'launch_provenance_verified': False, 'telemetry_contents_qualified': False,
              'sequence_coverage_verified': False, 'benchmark_outputs_qualified': False,
              'scope': 'Lifecycle composition only; caller must independently supervise this worker. '
                       'Polling output limits can overshoot; these are not filesystem quotas. '
                       'Not full telemetry verification, frozen launch provenance or acceptance.'}
    with (directory / 'collection.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    return result
