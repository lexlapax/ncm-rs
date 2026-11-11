"""Fixed, bounded read-only trace sequence primitive; NOT a launch-ready collector.

Caller must supply reviewed commands/environments after identity, affinity and
availability preflight. Telemetry and whole-run supervision are not provided.
There is deliberately no CLI. This module contains no GPU or clock commands.
"""
import json
from pathlib import Path
import shutil
import time

from bounded_child import run_bounded

ORDER = ('AP', 'CP', 'AT', 'CT', 'CP', 'AT', 'CT', 'AP', 'AT', 'CT', 'AP', 'CP')
CHILD_SECONDS = 120
SEQUENCE_SECONDS = 600
CAPTURE_BYTES = 1024 ** 3
TOTAL_BYTES = 6 * CAPTURE_BYTES
MIN_FREE_BYTES = 12 * CAPTURE_BYTES


def validate_jobs(jobs):
    """Copy/validate every job before any output creation or child launch.

    This checks structure and fixed order, NOT executable/source identity or
    injection isolation. Environment contents are deliberately not serialized.
    """
    if not isinstance(jobs, (list, tuple)) or len(jobs) != len(ORDER):
        raise ValueError('exactly twelve jobs required')
    validated = []
    for expected, job in zip(ORDER, jobs):
        if not isinstance(job, dict) or set(job) not in (
                {'condition', 'argv', 'cwd', 'env'},
                {'condition', 'argv', 'cwd', 'env', 'report_path'}):
            raise ValueError('invalid job fields')
        if job['condition'] != expected:
            raise ValueError('fixed AP/CP/AT/CT order required')
        argv, env = job['argv'], job['env']
        if not isinstance(argv, (list, tuple)) or not argv or not all(
                isinstance(a, str) and a and '\0' not in a for a in argv):
            raise ValueError('invalid argv')
        if not isinstance(env, dict) or not all(
                isinstance(k, str) and k and '=' not in k and '\0' not in k
                and isinstance(v, str) and '\0' not in v for k, v in env.items()):
            raise ValueError('explicit valid environment required')
        cwd = Path(job['cwd']).resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError('cwd must be a directory')
        copied = dict(condition=expected, argv=list(argv), cwd=str(cwd), env=dict(env))
        if 'report_path' in job:
            path = Path(job['report_path'])
            if not path.is_absolute() or str(path) not in argv:
                raise ValueError('report must be an explicit absolute argv argument')
            copied['report_path'] = str(path)
        validated.append(copied)
    return validated


def run_sequence(jobs, output_dir, *, cancel_check=None):
    """Run the fixed sequence once; any failure stops later jobs without retry.

    Each merged stdout/stderr capture (plain too) is capped at 1 GiB; the total
    cap is 6 GiB. Cleanup time may extend the 600-second submission/capture budget.
    Failures outside the owned-child primitive leave partial evidence and propagate.
    Collector death/escaped descendants require separate supervision before use.
    """
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError('cancel_check must be callable')
    jobs = validate_jobs(jobs)
    output_dir = Path(output_dir)
    for ordinal, job in enumerate(jobs, 1):
        if 'report_path' in job:
            expected = output_dir.resolve() / f'{ordinal:02d}-{job["condition"]}.report.json'
            if Path(job['report_path']) != expected:
                raise ValueError('report must use its unique sequence-owned path')
    # Parent must exist; do not implicitly create an arbitrary directory tree.
    if shutil.disk_usage(output_dir.parent).free < MIN_FREE_BYTES:
        raise ValueError('at least 12 GiB free required before sequence')
    started = time.monotonic()
    deadline = started + SEQUENCE_SECONDS
    output_dir.mkdir()  # exclusive: never reuse a partial or completed sequence
    manifest = {
        'schema_version': 1, 'diagnostic_only': True,
        'launch_readiness_verified': False,
        'sequence_seconds': SEQUENCE_SECONDS, 'child_seconds': CHILD_SECONDS,
        'capture_limit_bytes': CAPTURE_BYTES, 'total_limit_bytes': TOTAL_BYTES,
        'minimum_initial_free_bytes': MIN_FREE_BYTES,
        'jobs': [dict(ordinal=i, condition=j['condition'], argv=j['argv'], cwd=j['cwd'],
                      report_path=j.get('report_path'))
                 for i, j in enumerate(jobs, 1)],
    }
    with (output_dir / 'sequence-plan.json').open('x') as file:
        json.dump(manifest, file, indent=2)
        file.write('\n')
    used = 0
    receipts = []
    stop_reason = None
    for ordinal, job in enumerate(jobs, 1):
        if cancel_check is not None and cancel_check():
            stop_reason = 'external_cancelled'
            break
        if time.monotonic() >= deadline:
            stop_reason = 'sequence_deadline'
            break
        remaining = TOTAL_BYTES - used
        if remaining <= 0:
            stop_reason = 'aggregate_output_limit'
            break
        # Preserve the original headroom as our own bounded captures accumulate.
        # Concurrent external disk use can still cause write failure between checks.
        if shutil.disk_usage(output_dir).free < MIN_FREE_BYTES - used:
            stop_reason = 'free_space'
            break
        prefix = f'{ordinal:02d}-{job["condition"]}'
        receipt = run_bounded(
            job['argv'], cwd=job['cwd'], env=job['env'],
            capture_path=output_dir / (prefix + '.capture'),
            receipt_path=output_dir / (prefix + '.receipt.json'),
            timeout_seconds=CHILD_SECONDS, sequence_deadline=deadline,
            output_limit_bytes=min(CAPTURE_BYTES, remaining), report_path=job.get('report_path'),
            cancel_check=cancel_check)
        used += receipt['captured_bytes']
        receipts.append(dict(ordinal=ordinal, condition=job['condition'], receipt=receipt))
        if receipt['status'] != 'complete':
            stop_reason = 'child_failed'
            break
    result = {
        'schema_version': 1,
        'status': 'complete' if len(receipts) == len(ORDER) and stop_reason is None else 'failed',
        'stop_reason': stop_reason, 'captured_bytes': used,
        'elapsed_seconds_including_cleanup': time.monotonic() - started,
        'jobs': receipts, 'diagnostic_only': True,
        'launch_provenance_verified': False, 'benchmark_outputs_qualified': False,
        'whole_run_supervision_provided': False,
    }
    with (output_dir / 'sequence-result.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    return result
