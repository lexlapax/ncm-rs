import argparse
import json
import math
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description='Summarize completed three-process XSum CUDA comparisons.')
parser.add_argument('--directory', type=Path, default=Path('reports/phase4-xsum-cuda-performance-b32-v2'))
parser.add_argument('--batches', type=int, nargs='+', default=[32])
args = parser.parse_args()
root = args.directory
summary = {'batches': {}}
for batch in args.batches:
    result = summary['batches'][str(batch)] = {}
    for runtime in ('rust', 'python'):
        values = [json.loads((root / f'{runtime}-b{batch}-r{i}.json').read_text()) for i in range(3)]
        samples = [n * 1000 for value in values for n in value['trials_seconds']]
        assert len(samples) == 90
        times = [(root / f'{runtime}-b{batch}-r{i}.time').read_text() for i in range(3)]
        rss = [int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', text).group(1)) / 1024 for text in times]
        result[runtime] = {
            'process_medians_ms': [statistics.median(v['trials_seconds']) * 1000 for v in values],
            'pooled_median_ms': statistics.median(samples),
            'median_of_process_medians_ms': statistics.median(statistics.median(v['trials_seconds']) * 1000 for v in values),
            'pooled_p95_ms': sorted(samples)[math.ceil(.95 * len(samples)) - 1],
            'rss_mib': rss,
            'device': [v['cuda_default_pool'] if runtime == 'rust' else
                       {'peak_allocated_bytes': v['peak_allocated_bytes'], 'peak_reserved_bytes': v['peak_reserved_bytes']}
                       for v in values],
        }
        if runtime == 'python':
            cleanup = [n * 1000 for value in values for n in value['cleanup_seconds']]
            both = [(a + b) * 1000 for value in values for a, b in zip(value['trials_seconds'], value['cleanup_seconds'])]
            assert len(cleanup) == len(both) == 90
            result[runtime].update({'cleanup_median_ms': statistics.median(cleanup),
                                    'inclusive_median_ms': statistics.median(both),
                                    'inclusive_p95_ms': sorted(both)[math.ceil(.95 * len(both)) - 1]})
events = [json.loads(line) for line in (root / 'process-events.jsonl').read_text().splitlines()]
assert len(events) == 12 * len(args.batches) and all(e.get('returncode', 0) == 0 for e in events)
parents = {e['pid'] for e in events if e['event'] == 'start'}
activity = [json.loads(line) for line in (root / 'activity.jsonl').read_text().splitlines()]
failures = sum(q.get('status') != 0 or 'error' in q for row in activity for q in row['queries'])
other_counts, other_memory, utilization, temperature, clocks = [], [], [], [], []
for row in activity:
    gpu = row['queries'][0].get('stdout', '').strip().split(',')
    if len(gpu) == 7:
        try:
            utilization.append(int(gpu[1].strip()))
            temperature.append(float(gpu[3].strip()))
            clocks.append(int(gpu[5].strip()))
        except ValueError:
            pass
    count = memory = 0
    for entry in row['queries'][1].get('stdout', '').splitlines():
        parts = entry.split(',')
        if len(parts) < 3:
            continue
        try:
            pid, mib = int(parts[0].strip()), int(parts[-1].strip())
            ppid = row.get('parents', {}).get(str(pid))
            if pid not in parents and ppid not in parents:
                count += 1
                memory += mib
        except ValueError:
            continue
    other_counts.append(count); other_memory.append(memory)
summary['activity'] = {'samples': len(activity), 'failed_queries': failures,
                       'other_process_count_range': [min(other_counts), max(other_counts)],
                       'other_process_memory_mib_range': [min(other_memory), max(other_memory)],
                       'gpu_utilization_percent_range': [min(utilization), max(utilization)],
                       'gpu_temperature_c_range': [min(temperature), max(temperature)],
                       'sm_clock_mhz_range': [min(clocks), max(clocks)]}
print(json.dumps(summary, indent=2))
