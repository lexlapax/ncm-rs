"""Validate source-derived CUPTI host-event boundaries; NOT kernel attribution.

Only DRIVER records are interpreted. Passing this check does not qualify outputs,
validate kernel/copy records, prove capture completeness, or establish causality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


DRIVER = re.compile(
    r'^DRIVER \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", '
    r'cbid (\d+), processId (\d+), threadId (\d+), correlationId (\d+)$'
)
LAYOUTS = {
    # All queries include stage profiling; only the prefix has two events/query.
    'codec-bench-b1': (2008, 145),
    'historical-block4': (36, 36),
}
ELAPSED = {'cuEventElapsedTime', 'cuEventElapsedTime_v2'}
SCOPES = ('encode', 'decode', 'reconstruct', 'resident_decoder_block')


def label(layout, ordinal):
    """Source-derived labels, not semantic verification of the recorded work."""
    if layout == 'historical-block4':
        batch, scope, offset = 4, 'resident_decoder_block', ordinal - 1
    elif ordinal == 1:
        return {'batch': 1, 'scope': 'reconstruct', 'phase': 'cold', 'index': 0}
    else:
        batch = 1
        scope = SCOPES[(ordinal - 2) // 36]
        offset = (ordinal - 2) % 36
    if offset == 0:
        phase, index = 'first', 0
    elif offset <= 5:
        phase, index = 'warmup', offset - 1
    else:
        phase, index = 'trial', offset - 6
    return {'batch': batch, 'scope': scope, 'phase': phase, 'index': index}


def validate_boundaries(raw, layout):
    if layout not in LAYOUTS:
        raise ValueError('unsupported source layout')
    total_queries, prefix_queries = LAYOUTS[layout]
    text = raw.decode('utf-8')
    if 'dropped' in text.lower():
        raise ValueError('capture reports dropped records')
    records = []
    correlations = set()
    identities = set()
    for line in text.splitlines():
        if not line.startswith('DRIVER'):
            continue
        match = DRIVER.fullmatch(line)
        if match is None:
            raise ValueError('malformed DRIVER record')
        start, end, duration, name, cbid, process, thread, correlation = match.groups()
        start, end, duration = int(start), int(end), int(duration)
        if end < start or end - start != duration:
            raise ValueError('invalid DRIVER interval/duration')
        identity = (int(process), int(thread))
        identities.add(identity)
        correlation = int(correlation)
        if correlation in correlations:
            raise ValueError('duplicate DRIVER correlation ID')
        correlations.add(correlation)
        if name.startswith('cuEventRecord') and name != 'cuEventRecord':
            raise ValueError('unsupported event-record API')
        if name.startswith('cuEventElapsedTime') and name not in ELAPSED:
            raise ValueError('unsupported elapsed-time API')
        if name == 'cuEventRecord' or name in ELAPSED:
            records.append({'start_ns': start, 'end_ns': end,
                            'api': name, 'correlation_id': correlation})
    if len(identities) != 1:
        raise ValueError('requires exactly one DRIVER process/thread')
    # CUPTI prints kernels and APIs interleaved; printed order is not time order.
    records.sort(key=lambda row: row['start_ns'])
    for previous, current in zip(records, records[1:]):
        if (current['start_ns'] <= previous['start_ns']
                or current['start_ns'] < previous['end_ns']):
            raise ValueError('ambiguous/overlapping host event boundaries')
    queries = [row for row in records if row['api'] in ELAPSED]
    events = [row for row in records if row['api'] == 'cuEventRecord']
    if len(queries) != total_queries or len(events) != 2 * total_queries:
        raise ValueError('unexpected query/event count for source layout')
    if records[-1]['api'] not in ELAPSED:
        raise ValueError('event record after final elapsed-time query')
    pending = []
    windows = []
    for record in records:
        if record['api'] == 'cuEventRecord':
            pending.append(record)
            continue
        if len(pending) != 2:
            raise ValueError('prefix query does not have exactly two event records')
        ordinal = len(windows) + 1
        windows.append({'ordinal': ordinal, 'assumed_work': label(layout, ordinal),
                        'start_event_host': pending[0], 'end_event_host': pending[1],
                        'elapsed_query_host': record})
        pending = []
        if ordinal == prefix_queries:
            break
    process, thread = next(iter(identities))
    return {
        'schema_version': 1, 'trace_sha256': hashlib.sha256(raw).hexdigest(),
        'assumed_source_layout': layout, 'process_id': process, 'thread_id': thread,
        'total_elapsed_queries': len(queries), 'validated_prefix_windows': len(windows),
        'scope': 'Host boundary structure only; source-derived labels are not kernel '
                 'attribution, output qualification, or proof of complete capture.',
        'windows': windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layout', required=True, choices=LAYOUTS)
    args = parser.parse_args()
    result = validate_boundaries(args.trace.read_bytes(), args.layout)
    result['validator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
