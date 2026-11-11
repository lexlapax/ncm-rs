"""Read-only observer lifetime/readiness worker; not a diagnostic launch command.

Caller must supervise this dedicated worker and supply frozen commands/environment.
No benchmark is launched. Readiness and completion structure are not independent
provenance, full telemetry qualification or sequence coverage verification.
"""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from bounded_child import run_bounded
from readonly_trace_preflight import require
from run_phase2_linux_comparison import observer_affinity

SECONDS = 900
STARTUP_SECONDS = 15
CPU_TIMEOUT_SECONDS = 930
FILE_LIMIT = 16 * 1024 ** 2
WORKLOAD_CPU = 2
OBSERVER_CPU = 0
GPU_HEADER = ['timestamp', 'uuid', 'utilization.gpu [%]', 'memory.used [MiB]',
              'temperature.gpu', 'power.draw [W]', 'clocks.current.sm [MHz]',
              'clocks.current.memory [MHz]']


def commands(root, directory, python, taskset, nvidia_smi, gpu_uuid):
    """Construct fixed-duration read-only commands; no probing or execution."""
    root, directory = Path(root).absolute(), Path(directory).absolute()
    require(isinstance(gpu_uuid, str) and gpu_uuid.startswith('GPU-')
            and len(gpu_uuid) == 40 and all(c in '0123456789abcdef-' for c in gpu_uuid[4:].lower()),
            'expected a preverified GPU UUID')
    for path in (python, taskset, nvidia_smi):
        require(Path(path).is_absolute(), 'absolute frozen utility paths required')
    # The sampler checks CPU2 eligibility BEFORE pinning itself to CPU0. Give it
    # startup affinity {0,2}; its first metadata/sample must confirm final {0}.
    cpu = [str(taskset), '-c', f'{OBSERVER_CPU},{WORKLOAD_CPU}', str(python),
           str(root / 'tools/bench/sample_linux_cpu.py'), str(directory / 'cpu.jsonl'),
           '--cpu', str(WORKLOAD_CPU), '--observer-cpu', str(OBSERVER_CPU),
           '--seconds', str(SECONDS), '--interval', '1']
    gpu = [str(nvidia_smi), '-i', gpu_uuid,
           '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem',
           '--format=csv', '-l', '1']
    return cpu, gpu


def first_lines(path, count=2):
    if not path.exists():
        return None
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= FILE_LIMIT,
            'observer output type/size violation')
    with path.open('rb') as file:
        lines = [file.readline(65537) for _ in range(count)]
    require(all(len(line) <= 65536 for line in lines), 'observer prefix line too large')
    if not all(line.endswith(b'\n') for line in lines):
        return None
    return [line.decode('utf-8') for line in lines]


def validate_cpu_prefix(lines, expected_source):
    metadata, first = [json.loads(line) for line in lines]
    require(metadata['kind'] == 'metadata' and type(metadata['schema_version']) is int
            and metadata['schema_version'] == 1
            and metadata['source_sha256'] == expected_source
            and metadata['target_cpu'] == WORKLOAD_CPU
            and metadata['observer_affinity'] == [OBSERVER_CPU]
            and metadata['seconds'] == SECONDS and metadata['interval_seconds'] == 1,
            'CPU telemetry readiness mismatch')
    require(first['kind'] == 'sample' and type(first['index']) is int and first['index'] == 0,
            'missing first CPU sample')
    return metadata


def validate_gpu_prefix(lines, gpu_uuid):
    header, row = list(csv.reader(lines, skipinitialspace=True))
    require(header == GPU_HEADER and len(row) == len(header) and row[1] == gpu_uuid,
            'GPU telemetry readiness mismatch')
    datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f')
    return row


def stop_owned_gpu(child):
    """Term-before-reap, bounded kill escalation; never signal after successful wait."""
    try:
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        child.poll()  # settle Popen bookkeeping, never interpret synthetic ECHILD status
        return {'verified': False, 'reason': 'ownership_unconfirmed'}
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    escalated = False
    try:
        code = child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        # Timeout does not reap; revalidate ownership before the final group signal.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        os.killpg(child.pid, signal.SIGKILL)
        escalated = True
        code = child.wait(timeout=2)
    return {'verified': True, 'returncode': code, 'kill_escalation': escalated,
            'scope': 'Intentional observer shutdown, not benchmark success or complete GPU telemetry.'}


def collect(cpu_argv, gpu_argv, *, cwd, env, output_dir, expected_cpu_source, gpu_uuid):
    """Wait for full CPU sampler; stop GPU observer afterward, preserving failures.

    Directory is exclusive. ready.json appears only after both first samples.
    The caller must not start benchmarks before it and must cancel on worker loss.
    """
    require(sys.platform == 'linux', 'Linux observers required')
    require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
            'single-threaded observer with exclusive reaping/default SIGCHLD required')
    require(not any(k.startswith(('LD_', 'CUDA_', 'CUPTI_', 'NVTX_')) and v for k, v in env.items()),
            'observer injection/control environment forbidden')
    directory = Path(output_dir).absolute()
    cpu_path, gpu_path = directory / 'cpu.jsonl', directory / 'gpu.csv'
    require(str(cpu_path) in cpu_argv, 'CPU sampler output must be sequence-owned')
    directory.mkdir(mode=0o700)
    gpu = None
    ready = False
    errors = []
    cpu_receipt = None
    gpu_cleanup = None
    interrupted = None
    saved = {}
    start = time.monotonic()

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def observe(allow_readiness=True):
        nonlocal ready
        if interrupted is not None:
            return True
        if os.waitid(os.P_PID, gpu.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            errors.append('GPU_observer_exited_early')
            return True
        for path in (cpu_path, gpu_path):
            if path.exists() and (path.is_symlink() or not path.is_file() or path.stat().st_size > FILE_LIMIT):
                errors.append('observer_output_limit_or_type')
                return True
        if not ready:
            if not allow_readiness:
                errors.append('CPU_finished_before_readiness')
                return True
            if time.monotonic() - start >= STARTUP_SECONDS:
                errors.append('observer_startup_deadline')
                return True
            try:
                cpu_lines, gpu_lines = first_lines(cpu_path), first_lines(gpu_path)
                if cpu_lines is not None and gpu_lines is not None:
                    metadata = validate_cpu_prefix(cpu_lines, expected_cpu_source)
                    first_gpu = validate_gpu_prefix(gpu_lines, gpu_uuid)
                    record = {'schema_version': 1, 'kind': 'telemetry_ready', 'worker_pid': os.getpid(),
                              'gpu_pid': gpu.pid, 'cpu_metadata': metadata, 'first_gpu_row': first_gpu,
                              'observer_affinity': sorted(os.sched_getaffinity(0)),
                              'local_time_with_offset': datetime.now().astimezone().isoformat()}
                    pending = directory / 'ready.pending.json'
                    with pending.open('x') as file:
                        json.dump(record, file, indent=2)
                        file.write('\n')
                    # Validation/writing may consume the startup allowance too.
                    if time.monotonic() - start >= STARTUP_SECONDS:
                        errors.append('observer_startup_deadline')
                        return True
                    # Publish a complete marker atomically without overwriting one.
                    os.link(pending, directory / 'ready.json')
                    pending.unlink()
                    ready = True
            except (ValueError, KeyError, OSError) as error:
                errors.append('readiness_error:' + type(error).__name__)
                return True
        return False

    with observer_affinity(OBSERVER_CPU), gpu_path.open('xb', buffering=0) as gpu_output:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.signal(sig, interrupt)
            if interrupted is not None:
                errors.append('interrupted_before_observer_spawn')
            else:
                gpu = subprocess.Popen(gpu_argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=gpu_output, stderr=subprocess.STDOUT, start_new_session=True)
                cpu_receipt = run_bounded(cpu_argv, cwd=cwd, env=env,
                                          capture_path=directory / 'cpu-command.txt',
                                          receipt_path=directory / 'cpu-command.receipt.json',
                                          timeout_seconds=CPU_TIMEOUT_SECONDS, output_limit_bytes=1024 ** 2,
                                          report_path=cpu_path, report_limit_bytes=FILE_LIMIT,
                                          cancel_check=observe)
                # Never publish new readiness after a failed CPU capture.
                if cpu_receipt['status'] == 'complete':
                    observe(allow_readiness=False)
        except BaseException as error:
            errors.append('telemetry_error:' + type(error).__name__)
        finally:
            if gpu is not None:
                try:
                    gpu_cleanup = stop_owned_gpu(gpu)
                except BaseException as error:
                    errors.append('GPU_cleanup_error:' + type(error).__name__)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    if cpu_receipt is not None and cpu_receipt['status'] == 'complete':
        try:
            rows = [json.loads(line) for line in cpu_path.read_text().splitlines()]
            validate_cpu_prefix([json.dumps(row) for row in rows[:2]], expected_cpu_source)
            require(len(rows) == SECONDS + 3 and rows[-1]['kind'] == 'completion'
                    and rows[-1]['status'] == 'complete' and rows[-1]['samples'] == SECONDS + 1
                    and type(rows[-1]['actual_elapsed_seconds']) in (int, float)
                    and math.isfinite(rows[-1]['actual_elapsed_seconds'])
                    and rows[-1]['actual_elapsed_seconds'] >= SECONDS
                    and cpu_receipt['elapsed_seconds_including_cleanup'] >= SECONDS,
                    'incomplete CPU telemetry duration/inventory')
            require(all(row['kind'] == 'sample' and type(row['index']) is int and row['index'] == i
                        for i, row in enumerate(rows[1:-1])), 'CPU telemetry sample order mismatch')
        except (ValueError, KeyError, OSError) as error:
            errors.append('CPU_completion_error:' + type(error).__name__)
    if gpu_cleanup is not None and gpu_cleanup['verified'] and (
            gpu_cleanup['kill_escalation'] or gpu_cleanup['returncode'] not in (0, -signal.SIGTERM)):
        errors.append('GPU_shutdown_not_clean')
    complete = (ready and cpu_receipt is not None and cpu_receipt['status'] == 'complete'
                and gpu_cleanup is not None and gpu_cleanup['verified'] and not errors and interrupted is None)
    result = {'schema_version': 1, 'status': 'complete' if complete else 'failed', 'ready': ready,
              'cpu_receipt': cpu_receipt, 'gpu_shutdown': gpu_cleanup, 'errors': errors,
              'interrupt_signal': interrupted, 'telemetry_contents_qualified': False,
              'sequence_coverage_verified': False, 'launch_provenance_verified': False,
              'scope': 'Observer lifecycle and first-sample/CPU completion structure only. No '
                       'recomputed counters, whole-GPU-CSV validation, workload coverage or provenance.'}
    # Hash only bounded files; late GPU writes can exceed the polling boundary.
    if gpu_cleanup is not None and gpu_cleanup['verified']:
        with gpu_path.open('rb') as file:
            raw = file.read(FILE_LIMIT + 1)
        if len(raw) <= FILE_LIMIT:
            result['gpu_sha256'] = hashlib.sha256(raw).hexdigest()
        else:
            result['status'] = 'failed'
            result['errors'].append('final_GPU_output_limit')
    with (directory / 'telemetry.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    return result
