"""Read-only Linux CPU/environment telemetry; not a benchmark or acceptance gate.

No process names/command lines are collected. Optional affinity changes apply only
to this observer. JSONL ends with a completion record; interrupted files remain
incomplete. One-second frequency/utilization samples cannot identify per-call
clocks or establish that an entire future benchmark window will be quiet.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

TICK_NAMES = ('user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal')


def cpu_list(text):
    result = set()
    for part in text.strip().split(','):
        bounds = part.split('-')
        if len(bounds) not in (1, 2) or not all(b.isdecimal() for b in bounds):
            raise ValueError('invalid CPU list')
        start, end = int(bounds[0]), int(bounds[-1])
        if not 0 <= start <= end <= 65535:
            raise ValueError('invalid CPU range')
        result.update(range(start, end + 1))
    return sorted(result)


def parse_stat(text):
    cpus, counters = {}, {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        key = fields[0]
        if key == 'cpu' or (key.startswith('cpu') and key[3:].isdecimal()):
            if key in cpus or len(fields) < 9:
                raise ValueError('duplicate/incomplete CPU counters')
            values = [int(v) for v in fields[1:9]]
            if any(v < 0 for v in values):
                raise ValueError('negative CPU counter')
            # guest/guest_nice are already included in user/nice; do not add twice.
            cpus[key] = dict(zip(TICK_NAMES, values))
        elif key in ('ctxt', 'procs_running', 'procs_blocked'):
            if key in counters or len(fields) != 2 or int(fields[1]) < 0:
                raise ValueError('invalid system counter')
            counters[key] = int(fields[1])
    if 'cpu' not in cpus or set(counters) != {'ctxt', 'procs_running', 'procs_blocked'}:
        raise ValueError('incomplete /proc/stat')
    return cpus, counters


def activity(before, after):
    """Descriptive fractions of first-eight /proc/stat ticks, not CPU work units."""
    result = {}
    for cpu in before.keys() & after.keys():
        delta = {key: after[cpu][key] - before[cpu][key] for key in TICK_NAMES}
        if any(v < 0 for v in delta.values()):
            result[cpu] = {'available': False, 'reason': 'counter decreased or CPU changed'}
            continue
        total = sum(delta.values())
        result[cpu] = {'available': total > 0, 'total_ticks': total,
                       'non_idle_non_iowait_percent':
                           100 * (total - delta['idle'] - delta['iowait']) / total if total else None}
    return result


def optional_text(path):
    try:
        return {'available': True, 'value': path.read_text().strip()}
    except OSError as error:
        return {'available': False, 'error': type(error).__name__}


def snapshot(proc, sys_cpu, monitored):
    start = time.monotonic_ns()
    utc = datetime.now(timezone.utc).isoformat()
    ticks, counters = parse_stat((proc/'stat').read_text())
    if any(f'cpu{cpu}' not in ticks for cpu in monitored):
        raise ValueError('monitored CPU is offline or missing')
    frequencies = {str(cpu): optional_text(sys_cpu/f'cpu{cpu}/cpufreq/scaling_cur_freq') for cpu in monitored}
    loadavg = optional_text(proc/'loadavg')
    pressure = {kind: optional_text(proc/'pressure'/kind) for kind in ('cpu', 'memory', 'io')}
    return {'utc': utc, 'monotonic_start_ns': start, 'monotonic_end_ns': time.monotonic_ns(),
            'cpu_ticks': ticks, 'system_counters': counters,
            'frequency_khz': frequencies, 'loadavg': loadavg, 'pressure': pressure}


def run(output, cpu, seconds=60, interval=1, observer_cpu=None, proc=Path('/proc'),
        sys_cpu=Path('/sys/devices/system/cpu')):
    if platform.system() != 'Linux':
        raise ValueError('Linux telemetry only')
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (seconds, interval)):
        raise ValueError('finite sampling duration/interval required')
    if not 0.1 <= interval <= seconds <= 3600:
        raise ValueError('require 0.1 <= interval <= duration <= 3600 seconds')
    allowed = os.sched_getaffinity(0)
    if cpu not in allowed or (observer_cpu is not None and observer_cpu not in allowed):
        raise ValueError('CPU outside observer permitted affinity')
    siblings = cpu_list((sys_cpu/f'cpu{cpu}/topology/thread_siblings_list').read_text())
    if cpu not in siblings:
        raise ValueError('inconsistent SMT topology')
    if observer_cpu in siblings:
        raise ValueError('observer must not run on target or its SMT sibling')
    monitored = sorted(set(siblings) | ({observer_cpu} if observer_cpu is not None else set()))
    with output.open('x') as file:
        def emit(row):
            file.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n'); file.flush()
        if observer_cpu is not None:
            os.sched_setaffinity(0, {observer_cpu})
        emit({'kind': 'metadata', 'schema_version': 1, 'diagnostic_only': True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'platform': platform.platform(), 'target_cpu': cpu, 'smt_siblings': siblings,
              'observer_affinity': sorted(os.sched_getaffinity(0)), 'seconds': seconds, 'interval_seconds': interval,
              'governor': optional_text(sys_cpu/f'cpu{cpu}/cpufreq/scaling_governor'),
              'scaling_driver': optional_text(sys_cpu/f'cpu{cpu}/cpufreq/scaling_driver'),
              'tick_rate_hz': os.sysconf('SC_CLK_TCK'),
              'limitations': ['No quiet-window certification or acceptance decision.',
                              'CPU frequency is a driver-reported sample, not a per-request clock.',
                              'Non-idle/non-iowait includes steal time; guest ticks are not double counted.',
                              'Pressure, runnable tasks and context switches are system-wide, not per-request.',
                              'Observer overhead is not zero; raw read intervals and missed deadlines are retained.']})
        start = time.monotonic(); previous = None; index = 0; deadline = start
        while True:
            row = snapshot(proc, sys_cpu, monitored)
            row.update(kind='sample', index=index, scheduled_elapsed_seconds=deadline-start,
                       lateness_seconds=max(0, time.monotonic()-deadline))
            row['activity_since_previous'] = activity(previous, row['cpu_ticks']) if previous is not None else None
            emit(row); previous = row['cpu_ticks']; index += 1
            if deadline >= start + seconds:
                break
            deadline = min(deadline + interval, start + seconds)
            time.sleep(max(0, deadline-time.monotonic()))
        emit({'kind': 'completion', 'status': 'complete', 'samples': index,
              'actual_elapsed_seconds': time.monotonic()-start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--observer-cpu', type=int)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--interval', type=float, default=1)
    args = parser.parse_args()
    run(args.output, args.cpu, args.seconds, args.interval, args.observer_cpu)


if __name__ == '__main__':
    main()
