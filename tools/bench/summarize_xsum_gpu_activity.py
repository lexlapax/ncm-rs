"""Summarize sampled NVIDIA driver memory for owned XSum benchmark processes."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(manifest, events, samples):
    starts, ends = {}, {}
    for event in events:
        target = starts if event.get('event') == 'start' else ends if event.get('event') == 'end' else None
        if target is None or event['name'] in target:
            raise ValueError('invalid process events')
        target[event['name']] = event
    expected = {f'{runtime}-b{batch}-r{rep}' for batch in manifest['batches']
                for rep in range(manifest['processes_per_runtime_batch'])
                for runtime in ('rust', 'python')}
    if set(starts) != expected or set(ends) != expected:
        raise ValueError('incomplete process events')
    records = {}
    failed_queries = 0
    for sample in samples:
        query = sample['queries'][1]
        if query.get('status') != 0:
            failed_queries += 1
            continue
        for name in expected:
            start, end = starts[name], ends[name]
            if end.get('returncode') != 0 or not start['monotonic'] <= sample['monotonic'] <= end['monotonic']:
                continue
            for line in query.get('stdout', '').splitlines():
                pid, _, memory = line.partition(',')
                pid = pid.strip()
                if not pid.isdigit() or (int(pid) != start['pid'] and sample['parents'].get(pid) != start['pid']):
                    continue
                value = memory.rsplit(',', 1)[-1].strip().split()
                if (not value or not value[0].isdigit() or len(value) > 2
                        or (len(value) == 2 and value[1] != 'MiB')):
                    raise ValueError('invalid driver memory value')
                records.setdefault(name, []).append(int(value[0]))
    if set(records) != expected or any(ends[name].get('returncode') != 0 for name in expected):
        raise ValueError('missing samples or unsuccessful process')
    results = {}
    for batch in manifest['batches']:
        results[str(batch)] = {}
        for runtime in ('rust', 'python'):
            names = [f'{runtime}-b{batch}-r{rep}' for rep in range(manifest['processes_per_runtime_batch'])]
            peaks = [max(records[name]) for name in names]
            results[str(batch)][runtime] = {'sampled_peak_mib_per_process': peaks,
                                             'median_sampled_peak_mib': statistics.median(peaks),
                                             'samples_per_process': [len(records[name]) for name in names]}
    return {'scope': 'sampled NVIDIA driver memory of benchmark process only; peaks can be missed',
            'activity_samples': len(samples), 'failed_memory_queries': failed_queries, 'batches': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    event_bytes = (directory / 'process-events.jsonl').read_bytes()
    manifest = json.loads(manifest_bytes)
    events = [json.loads(line) for line in event_bytes.splitlines()]
    activity = (directory / 'activity.jsonl').read_bytes()
    samples = [json.loads(line) for line in activity.splitlines()]
    result = summarize(manifest, events, samples)
    result['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    result['process_events_sha256'] = hashlib.sha256(event_bytes).hexdigest()
    result['activity_sha256'] = hashlib.sha256(activity).hexdigest()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
