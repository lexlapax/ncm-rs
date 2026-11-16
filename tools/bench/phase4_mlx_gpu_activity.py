"""Read bounded, system-wide Metal accelerator snapshots on macOS.

These registry statistics are device-wide and do not measure target-process GPU time.
"""
import json
import plistlib
import subprocess
import time

FIELDS = ('Device Utilization %', 'Renderer Utilization %', 'Tiler Utilization %',
          'Alloc system memory', 'In use system memory')


def parse_snapshot(data):
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError('expected exactly one accelerator registry entry')
    entry = data[0]
    stats = entry.get('PerformanceStatistics')
    if not isinstance(stats, dict):
        raise ValueError('missing accelerator performance statistics')
    selected = {}
    for name in FIELDS:
        value = stats.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f'invalid accelerator statistic: {name}')
        selected[name] = value
    info = entry.get('AGCInfo', {})
    last_pid = info.get('fLastSubmissionPID') if isinstance(info, dict) else None
    if last_pid is not None and (type(last_pid) is not int or last_pid < 0):
        raise ValueError('invalid last-submission PID')
    return {'statistics': selected, 'last_submission_pid': last_pid}


def monitor_gpu(stop, path, errors, ready):
    try:
        with path.open('x') as out:
            while not stop.is_set():
                begin_mono = time.monotonic_ns()
                begin_wall = time.time_ns()
                result = subprocess.run(['/usr/sbin/ioreg', '-a', '-r', '-c', 'IOAccelerator', '-d', '1'],
                                        capture_output=True, timeout=10, check=True)
                if len(result.stdout) > 2 * 1024 * 1024:
                    raise ValueError('accelerator registry output exceeds bound')
                sample = parse_snapshot(plistlib.loads(result.stdout))
                sample.update({'begin_monotonic_ns': begin_mono, 'end_monotonic_ns': time.monotonic_ns(),
                               'begin_unix_ns': begin_wall, 'end_unix_ns': time.time_ns()})
                out.write(json.dumps(sample, sort_keys=True) + '\n')
                out.flush()
                ready.set()
                stop.wait(2)
    except Exception as error:
        errors.append(str(error))
    finally:
        ready.set()
