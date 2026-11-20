"""Summarize the retained Phase 5 WMT14 CFG=0 timing screen without outputs."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

BATCHES = (1, 4, 16, 32)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def summarize(manifest, measurements, events, activity):
    if (manifest['batches'] != list(BATCHES) or manifest['modes'] != [0]
            or manifest['repetitions'] != 3 or manifest['warmups'] != 5 or manifest['trials'] != 30):
        raise ValueError('unexpected workload')
    expected = {f'{runtime}-cfg0-b{batch}-r{rep}' for runtime in ('rust', 'python')
                for batch in BATCHES for rep in range(3)}
    starts, ends = {}, {}
    for event in events:
        table = starts if event.get('event') == 'start' else ends if event.get('event') == 'end' else None
        if table is None or event['name'] in table:
            raise ValueError('invalid events')
        table[event['name']] = event
    if set(starts) != expected or set(ends) != expected:
        raise ValueError('incomplete events')
    if any(ends[name]['returncode'] != 0 or ends[name]['monotonic'] < starts[name]['monotonic'] for name in expected):
        raise ValueError('unsuccessful process')
    keys = {f'cfg0-b{batch}-{runtime}' for batch in BATCHES for runtime in ('rust', 'python')}
    if set(measurements) != keys:
        raise ValueError('incomplete measurements')
    observed = {name: [] for name in expected}
    other_counts = []
    for sample in activity:
        queries = sample['queries']
        if len(queries) != 2 or any(q.get('status') != 0 or 'error' in q for q in queries):
            raise ValueError('GPU query failed')
        current = [name for name in expected if starts[name]['monotonic'] <= sample['monotonic'] <= ends[name]['monotonic']]
        if len(current) > 1:
            raise ValueError('overlapping processes')
        other = 0
        for line in queries[1]['stdout'].splitlines():
            pid, _, rest = line.partition(',')
            pid = pid.strip()
            if not pid.isdigit():
                raise ValueError('invalid GPU process')
            parts = rest.rsplit(',', 1)[-1].strip().split()
            if not parts or not parts[0].isdigit() or len(parts) > 2 or (len(parts) == 2 and parts[1] != 'MiB'):
                raise ValueError('invalid GPU memory')
            if current and (int(pid) == starts[current[0]]['pid'] or sample['parents'].get(pid) == starts[current[0]]['pid']):
                observed[current[0]].append(int(parts[0]))
            else:
                other += 1
        other_counts.append(other)
    if not activity or any(not values for values in observed.values()):
        raise ValueError('missing activity samples for a process')
    result = {'activity_samples': len(activity), 'other_process_count_range': [min(other_counts), max(other_counts)],
              'gpu_memory_scope': 'approximately 1 Hz sampled driver process memory; brief peaks may be missed', 'batches': {}}
    for batch in BATCHES:
        result['batches'][str(batch)] = {}
        for runtime in ('rust', 'python'):
            records = measurements[f'cfg0-b{batch}-{runtime}']
            if len(records) != 3:
                raise ValueError('missing repetitions')
            samples = [t * 1000 for record in records for t in record['trials_seconds']]
            if len(samples) != 90 or any(not math.isfinite(t) or t <= 0 for t in samples):
                raise ValueError('invalid timing trials')
            if any(len(record['device_event_seconds']) != 30 or record['peak_rss_kib'] <= 0 for record in records):
                raise ValueError('invalid device events or RSS')
            medians = [statistics.median(record['trials_seconds']) * 1000 for record in records]
            names = [f'{runtime}-cfg0-b{batch}-r{rep}' for rep in range(3)]
            result['batches'][str(batch)][runtime] = {
                'process_medians_ms': medians, 'median_of_process_medians_ms': statistics.median(medians),
                'pooled_p95_ms': sorted(samples)[math.ceil(.95 * len(samples)) - 1],
                'peak_rss_mib': [record['peak_rss_kib'] / 1024 for record in records],
                'sampled_gpu_peak_mib': [max(observed[name]) for name in names],
                'gpu_samples_per_process': [len(observed[name]) for name in names],
                'peak_allocated_bytes': [record['peak_allocated_bytes'] for record in records],
                'peak_reserved_bytes': [record['peak_reserved_bytes'] for record in records],
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    files = {name: (args.directory / name).read_bytes() for name in
             ('manifest.json', 'measurements.json', 'process-events.jsonl', 'activity.jsonl')}
    result = summarize(json.loads(files['manifest.json']), json.loads(files['measurements.json']),
                       [json.loads(line) for line in files['process-events.jsonl'].splitlines()],
                       [json.loads(line) for line in files['activity.jsonl'].splitlines()])
    result['input_sha256'] = {name: digest(data) for name, data in files.items()}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
