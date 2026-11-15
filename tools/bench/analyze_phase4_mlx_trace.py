"""Validate measured TinyStories signposts and count target GPU overlaps privately."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path):
    root = ET.parse(path).getroot()
    return root, {node.get('id'): node for node in root.iter() if node.get('id')}


def field(row, tag, index):
    node = row.find(tag)
    if node is not None and node.get('ref'):
        node = index[node.get('ref')]
    return node


def value(row, tag, index):
    node = field(row, tag, index)
    return '' if node is None else node.get('fmt') or node.text or ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    signposts = args.capture / 'signposts.xml'
    gpu = args.capture / 'gpu-intervals.xml'
    target = args.capture / 'target.json'
    data = json.loads(target.read_bytes())
    if data['batch'] != 128 or data['warmups'] != 5 or len(data['trials_seconds']) != 30:
        raise ValueError('unexpected target workload')
    root, index = parse(signposts)
    markers = defaultdict(list)
    process_names = set()
    for row in root.iter('row'):
        if value(row, 'subsystem', index) != 'org.ncm.phase2':
            continue
        description = value(row, 'os-log-metadata', index)
        match = re.fullmatch(r'batch=128 sequence=(\d+)', description)
        kind = value(row, 'event-type', index)
        name = value(row, 'signpost-name', index)
        if not match or kind not in ('Begin', 'End', 'Event'):
            raise ValueError('unexpected target signpost')
        if name != ('ncm_graph_end' if kind == 'Event' else 'ncm_block'):
            raise ValueError('unexpected target signpost name')
        process_names.add(value(row, 'process', index))
        timestamp = field(row, 'event-time', index)
        markers[(int(match[1]), kind)].append(int(timestamp.text))
    if len(process_names) != 1:
        raise ValueError('expected exactly one marked process')
    process_name = next(iter(process_names))
    if not re.fullmatch(r'tinystories_lifecycle_bench \(\d+\)', process_name):
        raise ValueError('unexpected marked process')
    windows = []
    marker_errors = []
    stream_count = None
    for sequence in range(5, 35):
        values = {}
        for kind in ('Begin', 'Event', 'End'):
            times = markers[(sequence, kind)]
            if not times or len(set(times)) != 1:
                raise ValueError(f'missing or inconsistent marker: {sequence}/{kind}')
            if stream_count is None:
                stream_count = len(times)
            if len(times) != stream_count:
                raise ValueError('inconsistent marker stream count')
            values[kind] = times[0]
        if not values['Begin'] <= values['Event'] <= values['End']:
            raise ValueError('markers are out of order')
        duration = (values['End'] - values['Begin']) / 1e9
        measured = data['trials_seconds'][sequence - 5]
        if not math.isfinite(measured) or measured <= 0:
            raise ValueError('invalid measured duration')
        error_ms = abs(duration - measured) * 1000
        if error_ms > 0.2:
            raise ValueError('marker and reported timing differ by more than 0.2 ms')
        marker_errors.append(error_ms)
        windows.append((sequence, values['Begin'], values['Event'], values['End']))
    root, index = parse(gpu)
    target_intervals = overlapping = 0
    overlapping_requests = set()
    clipped_intervals = defaultdict(list)
    for row in root.iter('row'):
        if value(row, 'process', index) != process_name:
            continue
        start, duration = field(row, 'start-time', index), field(row, 'duration', index)
        if start is None or duration is None:
            raise ValueError('target GPU interval lacks time or duration')
        begin = int(start.text)
        end = begin + int(duration.text)
        target_intervals += 1
        hit = False
        for sequence, left, _, right in windows:
            if begin < right and end > left:
                hit = True
                overlapping_requests.add(sequence)
                clipped_intervals[sequence].append((max(begin, left), min(end, right)))
        overlapping += int(hit)
    if not target_intervals or len(overlapping_requests) != 30:
        raise ValueError('GPU intervals do not overlap all measured requests')
    gpu_unions_ms = []
    for sequence, _, _, _ in windows:
        total, left, right = 0, 0, 0
        for begin, end in sorted(clipped_intervals[sequence]):
            if begin > right:
                total += right - left
                left, right = begin, end
            else:
                right = max(right, end)
        gpu_unions_ms.append((total + right - left) / 1e6)
    sample_phase_ms = [(event - begin) / 1e6 for _, begin, event, _ in windows]
    readback_phase_ms = [(end - event) / 1e6 for _, _, event, end in windows]
    trials = sorted(data['trials_seconds'])
    report = {'target_sha256': digest(target), 'signposts_sha256': digest(signposts),
              'gpu_intervals_sha256': digest(gpu), 'measured_requests': len(windows),
              'marker_streams': stream_count, 'max_marker_timing_error_ms': max(marker_errors),
              'warmup_zero_begin_present': bool(markers[(0, 'Begin')]),
              'target_gpu_intervals': target_intervals, 'intervals_overlapping_measured_requests': overlapping,
              'measured_requests_with_gpu_overlap': len(overlapping_requests),
              'median_sample_phase_ms': statistics.median(sample_phase_ms),
              'median_readback_cleanup_phase_ms': statistics.median(readback_phase_ms),
              'median_gpu_overlap_union_ms': statistics.median(gpu_unions_ms),
              'gpu_overlap_union_range_ms': [min(gpu_unions_ms), max(gpu_unions_ms)],
              'median_ms': statistics.median(trials) * 1000, 'p95_ms': trials[28] * 1000,
              'maximum_ms': trials[-1] * 1000, 'trials_over_4_seconds': sum(t > 4 for t in trials),
              'limitations': ['Instrumented measurements are not a performance gate.',
                              'GPU overlaps do not establish cause of a slow request.',
                              'GPU overlap is the union of target intervals clipped to request windows;',
                              'it is not a complete device-utilization or causal decomposition.']}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
