"""Align diagnostic request wall timestamps with system-wide GPU snapshots."""
import argparse
import json
import math
from pathlib import Path


def analyze(report, snapshots):
    intervals = report['request_wall_ns']
    times = report['trials_seconds']
    warmups = report['warmups']
    if warmups != 5 or len(times) != 30 or len(intervals) != warmups + len(times):
        raise ValueError('incomplete diagnostic request sequence')
    previous = 0
    for interval in intervals:
        if (not isinstance(interval, list) or len(interval) != 3
                or any(type(x) is not int for x in interval)
                or not previous <= interval[0] <= interval[1] <= interval[2]):
            raise ValueError('invalid or nonmonotone request timestamps')
        previous = interval[2]
    if not snapshots:
        raise ValueError('no GPU snapshots')
    last_sample = 0
    for sample in snapshots:
        begin, end = sample['begin_unix_ns'], sample['end_unix_ns']
        if type(begin) is not int or type(end) is not int or not last_sample <= begin <= end:
            raise ValueError('invalid or nonmonotone GPU snapshot timestamps')
        last_sample = end
    rows = []
    for index, (duration, interval) in enumerate(zip(times, intervals[warmups:])):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid request duration')
        begin, sample_done, end = interval
        if abs((end - begin) / 1e9 - duration) > 0.02:
            raise ValueError('wall and monotonic duration disagree by over 20 ms')
        observations = [s for s in snapshots if begin <= s['end_unix_ns'] <= end]
        rows.append({'trial': index, 'duration_ms': duration * 1000,
                     'sample_phase_ms': (sample_done - begin) / 1e6,
                     'gpu_snapshot_count': len(observations),
                     'reported_device_utilization': [s['statistics']['Device Utilization %'] for s in observations],
                     'last_submission_pids': [s['last_submission_pid'] for s in observations]})
    return {'requests': rows, 'requests_over_4_seconds': sum(row['duration_ms'] > 4000 for row in rows),
            'requests_without_gpu_snapshot': sum(row['gpu_snapshot_count'] == 0 for row in rows),
            'limitations': ['Device counters are system-wide snapshots with an uncalibrated averaging window.',
                            'Last-submission PID does not attribute GPU time or establish cause.',
                            'Wall-clock alignment is diagnostic and can be affected by clock adjustments.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu-activity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    report = json.loads(args.report.read_bytes())
    snapshots = [json.loads(line) for line in args.gpu_activity.read_text().splitlines()]
    args.output.write_text(json.dumps(analyze(report, snapshots), indent=2) + '\n')


if __name__ == '__main__':
    main()
