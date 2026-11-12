"""Bounded fixed codec worker; import-only, no GPU policy commands or launch entry.

Caller owes frozen identities, CPU0 placement, telemetry readiness/failure channel,
independent restoration supervision and fresh availability. This worker neither
changes clocks nor establishes hardware policy, provenance or numerical quality.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import time

import diagnostic_clock
from bounded_child import run_bounded
from run_phase2_linux_comparison import plan_jobs

WORK_SECONDS = 590
CAPTURE_BYTES = 1024 ** 2
REPORT_BYTES = 16 * 1024 ** 2
MIN_FREE_BYTES = 1024 ** 3


def run_sequence(previous, previous_binaries, output_dir, *, root, env,
                 work_deadline, cancel_check, process_observer_factory,
                 taskset='/usr/bin/taskset'):
    """Run six codec-only cyclic repetitions once; first failure stops submission.

    Deadline is supplied by the armed restorer, never restarted here. Reports and
    captures use separate runtime directories, retaining the existing report argv.
    Python copies are deferred to post-collection analysis, not independent jobs.
    All plans/env/observer objects are checked before output creation or spawning.
    Observer callbacks must be trusted/bounded and must not hash frozen files.
    """
    if not callable(cancel_check) or not callable(process_observer_factory):
        raise ValueError('mandatory cancellation and process observation required')
    if (type(work_deadline) not in (int, float) or not math.isfinite(work_deadline)
            or not time.monotonic() < work_deadline <= time.monotonic() + WORK_SECONDS):
        raise ValueError('live absolute work deadline at most 590 seconds away required')
    expected_env = {'PATH': os.defpath, 'LANG': 'C', 'LC_ALL': 'C',
                    'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    if (not isinstance(env, dict) or set(env) != set(expected_env) | {'HOME'}
            or any(env[k] != v for k, v in expected_env.items())
            or not isinstance(env['HOME'], str) or '\0' in env['HOME']
            or not Path(env['HOME']).is_absolute()):
        raise ValueError('explicit isolated environment required')
    env = dict(env)
    root, previous, previous_binaries = (Path(p).resolve(strict=True)
                                        for p in (root, previous, previous_binaries))
    if not all(p.is_dir() for p in (root, previous, previous_binaries)):
        raise ValueError('existing checkout/binary directories required')
    directory = Path(output_dir).absolute()
    if os.path.lexists(directory) or not directory.parent.is_dir():
        raise ValueError('fresh output directory with existing parent required')
    directory = directory.resolve()
    if not Path(taskset).is_absolute() or not Path(taskset).is_file() or not os.access(taskset, os.X_OK):
        raise ValueError('absolute executable taskset required')
    jobs = plan_jobs(previous, previous_binaries, directory, root=root,
                     repeats=6, workloads=('codec',), cpu=2, taskset=taskset)
    order = ('previous', 'python', 'candidate', 'python', 'candidate', 'previous',
             'candidate', 'previous', 'python') * 2
    if len(jobs) != 18:
        raise ValueError('exactly eighteen codec jobs required')
    for index, job in enumerate(jobs):
        runtime, repeat = order[index], index // 3 + 1
        kind = 'python' if runtime == 'python' else 'rust'
        report = str(directory / runtime / f'codec-{kind}-{repeat}.json')
        if (job['runtime'] != runtime or job['repeat'] != repeat or job['workload'] != 'codec'
                or job['report'] != report or job['argv'][-1] != report
                or job['argv'][:3] != [taskset, '-c', '2']):
            raise ValueError('fixed codec order/CPU/report paths required')
    observers = []
    for ordinal, job in enumerate(jobs, 1):
        if not os.access(job['argv'][3], os.X_OK) or not Path(job['argv'][3]).is_file():
            raise ValueError('every workload executable must exist before first spawn')
        observer = process_observer_factory(ordinal, deepcopy(job), dict(env))
        if (observer is None or not callable(getattr(observer, 'observe', None))
                or not callable(getattr(observer, 'finish', None))):
            raise ValueError('enabled process observation requires every observer')
        observers.append(observer)
    if shutil.disk_usage(directory.parent).free < MIN_FREE_BYTES:
        raise ValueError('at least 1 GiB initial free space required')
    lifetime = diagnostic_clock.begin()
    directory.mkdir(mode=0o700)
    for runtime in ('previous', 'candidate', 'python'):
        (directory / runtime).mkdir(mode=0o700)
    manifest = {'schema_version': 1, 'diagnostic_only': True, 'jobs': jobs,
                'work_deadline_monotonic_seconds': work_deadline,
                'capture_limit_bytes_per_job': CAPTURE_BYTES,
                'report_limit_bytes_per_job': REPORT_BYTES,
                'minimum_initial_free_bytes': MIN_FREE_BYTES,
                'python_copies_deferred': True, 'launch_readiness_verified': False}
    with (directory / 'sequence-plan.json').open('x') as file:
        json.dump(manifest, file, indent=2)
        file.write('\n')
    cancelled = False

    def cancellation():
        nonlocal cancelled
        if not cancelled:
            try:
                status = cancel_check()
                if type(status) is not bool:
                    raise ValueError('cancellation status must be Boolean')
                cancelled = status
            except BaseException:
                cancelled = True
                raise
        return cancelled

    receipts, stop_reason = [], None
    for ordinal, (job, observer) in enumerate(zip(jobs, observers), 1):
        receipt = observation = None
        try:
            if cancellation():
                stop_reason = 'external_cancelled'
                break
            if time.monotonic() >= work_deadline:
                stop_reason = 'work_deadline'
                break
            if shutil.disk_usage(directory).free < MIN_FREE_BYTES:
                stop_reason = 'free_space'
                break
            report = Path(job['report'])
            prefix = report.with_suffix('')
            receipt = run_bounded(
                job['argv'], cwd=job['cwd'], env=env,
                capture_path=prefix.with_suffix('.capture'),
                receipt_path=prefix.with_suffix('.receipt.json'),
                report_path=report, report_limit_bytes=REPORT_BYTES,
                timeout_seconds=WORK_SECONDS, sequence_deadline=work_deadline,
                output_limit_bytes=CAPTURE_BYTES, cancel_check=cancellation,
                observe_child=observer.observe)
            observation = observer.finish(receipt, prefix.with_suffix('.process.json'))
            if receipt['status'] != 'complete':
                stop_reason = 'child_failed'
            elif not isinstance(observation, dict) or observation.get('status') != 'complete':
                stop_reason = 'process_observation_failed'
            elif cancellation():
                stop_reason = 'external_cancelled'
        except BaseException as error:
            stop_reason = 'sequence_error:' + type(error).__name__
        receipts.append({'ordinal': ordinal, 'runtime': job['runtime'],
                         'repeat': job['repeat'], 'receipt': receipt, 'observation': observation})
        if stop_reason is not None:
            break
    result = {'schema_version': 1,
              'status': 'complete' if len(receipts) == 18 and stop_reason is None else 'failed',
              'stop_reason': stop_reason, 'jobs': receipts, 'external_cancelled': cancelled,
              'work_deadline_monotonic_seconds': work_deadline,
              'lifetime': diagnostic_clock.finish(lifetime), 'diagnostic_only': True,
              'python_copies_deferred': True, 'launch_provenance_verified': False,
              'benchmark_outputs_qualified': False, 'whole_run_supervision_provided': False}
    with (directory / 'sequence-result.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    return result
