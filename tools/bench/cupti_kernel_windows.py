"""Offline kernel/host-launch correlation inside source-assumed block windows.

Not output qualification or complete activity attribution. Kernel, copy and memset
records are correlated to host calls; bookkeeping/overhead activity stays separate.
Time not covered by these activities is not proven CPU overhead or GPU idle time.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

from cupti_trace_boundaries import DRIVER, validate_boundaries

RUNTIME = re.compile(DRIVER.pattern.replace('^DRIVER', '^RUNTIME'))
KERNEL = re.compile(
    r'^CONCURRENT_KERNEL \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", '
    r'correlationId (\d+), cacheConfigRequested \d+, cacheConfigExecuted \d+$'
)
DEVICE = re.compile(
    r'^\s+deviceId (\d+), contextId (\d+), streamId (\d+), '
    r'graphId (\d+), graphNodeId (\d+),.*$'
)
COPY = re.compile(
    r'^MEMCPY "([A-Z]+)" \[ (\d+), (\d+) \] duration (\d+), size (\d+), '
    r'copyCount (\d+), srcKind ([A-Z_]+), dstKind ([A-Z_]+), correlationId (\d+)$'
)
MEMSET = re.compile(
    r'^MEMSET \[ (\d+), (\d+) \] duration (\d+), value (\d+), size (\d+), correlationId (\d+)$'
)
COPY_APIS = {'HTOD': {'cuMemcpyHtoDAsync_v2'}, 'DTOH': {'cuMemcpyDtoHAsync_v2'},
             'DTOD': {'cuMemcpyDtoDAsync_v2'}}
MEMSET_APIS = {'cudaMemsetAsync_v3020', 'cuMemsetD8Async'}
MEMORY_APIS = set.union(*COPY_APIS.values(), MEMSET_APIS)
OTHER = re.compile(r'^([A-Z][A-Z0-9_]*).*? \[ (\d+)(?:, (\d+))? \] (.*)$')
LAUNCHES = {'cuLaunchKernel', 'cuLaunchKernelEx', 'cuLaunchKernel_ptsz', 'cuLaunchKernelEx_ptsz',
            'cudaLaunchKernel_v7000', 'cudaLaunchKernelExC_v11060'}


def union_ns(intervals):
    """Union, not sum: concurrent/overlapping intervals must not double-count."""
    total = 0
    right = None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('reversed interval')
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def intersects(start, end, left, right):
    return left <= start < right if start == end else start < right and end > left


def device_identity(lines, index):
    identities = []
    for continuation in lines[index + 1:index + 5]:
        if not continuation[:1].isspace():
            break
        identity = DEVICE.fullmatch(continuation)
        if identity:
            identities.append(tuple(map(int, identity.groups())))
        parent = re.search(r'sourceGraphId (\d+), sourceGraphNodeId (\d+)', continuation)
        if 'sourceGraph' in continuation and parent is None:
            raise ValueError('malformed source graph metadata')
        if parent and any(int(value) for value in parent.groups()):
            raise ValueError('source graph activity is unsupported')
    if len(identities) != 1:
        raise ValueError('missing/ambiguous device/context/stream')
    device, context, stream, graph, node = identities[0]
    if graph or node:
        raise ValueError('graph activity is unsupported')
    return {'device': device, 'context': context, 'stream': stream}


def parse_memory_activity(lines, index):
    line = lines[index]
    if line.startswith('MEMCPY'):
        match = COPY.fullmatch(line)
        if match is None:
            raise ValueError('unsupported/malformed copy record')
        direction, start, end, duration, size, count, source, target, corr = match.groups()
        if direction not in COPY_APIS or int(count) != 1:
            raise ValueError('unsupported copy direction/count')
        hosts = {'PAGEABLE', 'PINNED'}
        kinds_ok = ((direction == 'HTOD' and source in hosts and target == 'DEVICE')
                    or (direction == 'DTOH' and source == 'DEVICE' and target in hosts)
                    or (direction == 'DTOD' and source == target == 'DEVICE'))
        if not kinds_ok:
            raise ValueError('unsupported/inconsistent copy memory kinds')
        details = {'kind': 'MEMCPY', 'direction': direction, 'source_kind': source,
                   'destination_kind': target}
    else:
        match = MEMSET.fullmatch(line)
        if match is None:
            raise ValueError('unsupported/malformed memset record')
        start, end, duration, value, size, corr = match.groups()
        details = {'kind': 'MEMSET', 'value': int(value)}
    start, end, duration, size, corr = map(int, (start, end, duration, size, corr))
    if end <= start or duration != end - start or size <= 0:
        raise ValueError('invalid memory activity interval/size')
    return {**details, **device_identity(lines, index), 'start_ns': start, 'end_ns': end,
            'duration_ns': duration, 'bytes': size, 'correlation_id': corr}


def summarize(raw, layout):
    boundaries = validate_boundaries(raw, layout)
    lines = raw.decode('utf-8').splitlines()
    host_apis = {}
    kernels = {}
    memory = {}
    other = []
    for index, line in enumerate(lines):
        if line.startswith(('DRIVER', 'RUNTIME')):
            domain = line.split(' ', 1)[0]
            match = (DRIVER if domain == 'DRIVER' else RUNTIME).fullmatch(line)
            if match is None:
                raise ValueError('malformed host API record')
            start, end, duration, name, _, process, thread, corr = match.groups()
            start, end, duration, corr = map(int, (start, end, duration, corr))
            if end < start or end - start != duration:
                raise ValueError('invalid host API interval/duration')
            if (int(process), int(thread)) != (boundaries['process_id'], boundaries['thread_id']):
                raise ValueError('multiple host process/thread attribution is unsupported')
            if corr in host_apis:
                raise ValueError('ambiguous host API correlation')
            host_apis[corr] = {'start_ns': start, 'end_ns': end, 'domain': domain,
                             'duration_ns': duration, 'name': name, 'correlation_id': corr}
        elif line.startswith('CONCURRENT_KERNEL'):
            match = KERNEL.fullmatch(line)
            if match is None:
                raise ValueError('malformed kernel header')
            start, end, duration, name, corr = match.groups()
            start, end, duration, corr = map(int, (start, end, duration, corr))
            if end <= start or end - start != duration:
                raise ValueError('invalid kernel interval/duration')
            if corr in kernels:
                raise ValueError('duplicate kernel correlation')
            kernels[corr] = {'start_ns': start, 'end_ns': end, 'duration_ns': duration,
                             'name': name, 'correlation_id': corr, **device_identity(lines, index)}
        elif line.startswith(('MEMCPY', 'MEMSET')):
            activity = parse_memory_activity(lines, index)
            corr = activity['correlation_id']
            if corr in memory:
                raise ValueError('duplicate memory activity correlation')
            memory[corr] = activity
        elif line[:1].isupper() and ' [ ' in line:
            match = OTHER.fullmatch(line)
            if match is None:
                raise ValueError('unrecognized activity interval')
            kind, start, end, rest = match.groups()
            start = int(start)
            if end is None:
                end = start
            else:
                end = int(end)
                duration = re.match(r'duration (\d+)(?:,|$)', rest)
                if duration is None or end < start or end - start != int(duration[1]):
                    raise ValueError('invalid non-kernel activity interval/duration')
            other.append({'kind': kind, 'start_ns': start, 'end_ns': end, 'header': line})
        elif line.startswith(('MEMCPY', 'MEMSET', 'OVERHEAD', 'MEMORY2')):
            raise ValueError('malformed non-kernel activity')
    if kernels.keys() & memory.keys():
        raise ValueError('ambiguous kernel/memory correlation')
    for corr, activity in memory.items():
        host = host_apis.get(corr)
        supported = COPY_APIS[activity['direction']] if activity['kind'] == 'MEMCPY' else MEMSET_APIS
        if host is None or host['name'] not in supported:
            raise ValueError('memory activity has no supported host call')
        if activity['start_ns'] < host['start_ns']:
            raise ValueError('memory activity starts before host call')
    for corr, kernel in kernels.items():
        launch = host_apis.get(corr)
        if launch is None or launch['name'] not in LAUNCHES:
            raise ValueError('kernel has no unique supported host launch')
        if kernel['start_ns'] < launch['start_ns']:
            raise ValueError('kernel starts before host launch')

    rows = []
    stream_identities = set()
    for window in boundaries['windows']:
        if window['assumed_work']['scope'] != 'resident_decoder_block':
            continue
        left = window['start_event_host']['end_ns']
        right = window['end_event_host']['start_ns']
        if right <= left:
            raise ValueError('empty host window')
        selected = [row for row in host_apis.values()
                    if intersects(row['start_ns'], row['end_ns'], left, right)]
        if any(row['start_ns'] < left or row['end_ns'] > right for row in selected):
            raise ValueError('host API crosses window boundary')
        launch_ids = {row['correlation_id'] for row in selected if row['name'] in LAUNCHES}
        if not launch_ids or not launch_ids <= kernels.keys():
            raise ValueError('empty or unmatched kernel window')
        overlapping = {corr for corr, k in kernels.items()
                       if intersects(k['start_ns'], k['end_ns'], left, right)}
        if overlapping != launch_ids:
            raise ValueError('cross-window kernel correlation')
        sort_key = lambda row: (row['start_ns'], row['end_ns'], row['correlation_id'])
        selected_kernels = sorted((kernels[c] for c in launch_ids), key=sort_key)
        if any(k['start_ns'] < left or k['end_ns'] > right for k in selected_kernels):
            raise ValueError('kernel crosses synchronized window boundary')
        memory_calls = [row for row in selected if row['name'].startswith(
            ('cuMemcpy', 'cudaMemcpy', 'cuMemset', 'cudaMemset'))]
        if any(row['name'] not in MEMORY_APIS for row in memory_calls):
            raise ValueError('unsupported window memory API')
        memory_ids = {row['correlation_id'] for row in memory_calls}
        overlapping_memory = {corr for corr, activity in memory.items()
                              if intersects(activity['start_ns'], activity['end_ns'], left, right)}
        if not memory_ids <= memory.keys() or memory_ids != overlapping_memory:
            raise ValueError('missing or cross-window memory correlation')
        selected_memory = sorted((memory[c] for c in memory_ids), key=sort_key)
        if any(row['start_ns'] < left or row['end_ns'] > right for row in selected_memory):
            raise ValueError('memory activity crosses synchronized window boundary')
        device_activity = selected_kernels + selected_memory
        identities = {(k['device'], k['context'], k['stream']) for k in device_activity}
        stream_identities.update(identities)
        if len(stream_identities) != 1:
            raise ValueError('multiple device/context/stream attribution is unsupported')
        intervals = [(k['start_ns'], k['end_ns']) for k in selected_kernels]
        covered = union_ns(intervals)
        hull = max(end for _, end in intervals) - min(start for start, _ in intervals)
        device_intervals = [(a['start_ns'], a['end_ns']) for a in device_activity]
        device_covered = union_ns(device_intervals)
        device_hull = max(end for _, end in device_intervals) - min(start for start, _ in device_intervals)
        api_groups = collections.defaultdict(list)
        for row in selected:
            api_groups[row['name']].append(row['duration_ns'])
        rows.append({
            'ordinal': window['ordinal'], 'assumed_work': window['assumed_work'],
            'host_window_start_ns': left, 'host_window_end_ns': right,
            'kernel_sum_ns': sum(k['duration_ns'] for k in selected_kernels),
            'kernel_union_ns': covered, 'kernel_hull_ns': hull,
            'kernel_hull_uncovered_ns': hull - covered,
            'kernels': selected_kernels, 'memory_activity': selected_memory,
            'device_activity_union_ns': device_covered,
            'device_activity_hull_ns': device_hull,
            'device_activity_hull_uncovered_ns': device_hull - device_covered,
            'host_apis': sorted(selected, key=sort_key),
            'host_api_summary': {name: {'count': len(values), 'sum_ns': sum(values),
                                        'max_ns': max(values)}
                                 for name, values in sorted(api_groups.items())},
            'unattributed_activity': sorted(
                (row for row in other if intersects(row['start_ns'], row['end_ns'], left, right)),
                key=lambda row: (row['start_ns'], row['kind'], row['header'])),
        })
    return {
        'schema_version': 1, 'trace_sha256': boundaries['trace_sha256'],
        'assumed_source_layout': layout, 'activity_attribution_complete': False,
        'scope': 'Diagnostic kernel/copy/memset and host-call intervals only. Uncovered time is not '
                 'proven device idle or CPU overhead; bookkeeping/overhead activity is unattributed. '
                 'Source labels, output qualification and capture completeness remain separate.',
        'windows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layout', required=True, choices=('codec-bench-b1', 'historical-block4'))
    args = parser.parse_args()
    result = summarize(args.trace.read_bytes(), args.layout)
    result['source_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('cupti_kernel_windows.py', 'cupti_trace_boundaries.py')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
