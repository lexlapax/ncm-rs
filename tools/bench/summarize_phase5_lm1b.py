"""Aggregate repeated LM1B CUDA model probes without generated outputs."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from run_lm1b_small_capacity_screen import summarize

EXPECTED = {'small': (1, 4, 16), 'medium': (1, 4, 16), 'large': (1, 4, 8)}


def aggregate_run(manifest, result, activity):
    size = manifest['size']
    batches = tuple(manifest['batches'])
    if (size not in EXPECTED or not batches or any(batch not in EXPECTED[size] for batch in batches)
            or len(set(batches)) != len(batches) or manifest['warmups'] != 5 or manifest['trials'] != 30
            or manifest['repetitions'] != 3 or manifest['rust_convolution_math'] != 'cuDNN graph and efficient attention'):
        raise ValueError('unexpected run protocol')
    rows = result['rows']
    if summarize(rows, trials=30, batches=batches) != result['summary']:
        raise ValueError('timing summary differs')
    if any(not math.isfinite(row['load_seconds']) or row['load_seconds'] <= 0 or row['peak_rss_kib'] <= 0
           or row['end_monotonic'] < row['start_monotonic'] for row in rows):
        raise ValueError('invalid load, RSS or timing')
    observed = {row['pid']: [] for row in rows}
    if len(observed) != len(rows):
        raise ValueError('duplicate process ID')
    other_counts = []
    for sample in activity:
        queries = sample['queries']
        if len(queries) != 2 or any(q.get('status') != 0 or 'error' in q for q in queries):
            raise ValueError('GPU telemetry failed')
        current = [row for row in rows if row['start_monotonic'] <= sample['monotonic'] <= row['end_monotonic']]
        if len(current) > 1:
            raise ValueError('overlapping process intervals')
        other = 0
        for line in queries[1]['stdout'].splitlines():
            pid, _, rest = line.partition(',')
            memory = rest.rsplit(',', 1)[-1].strip().split()
            if not pid.strip().isdigit() or not memory or not memory[0].isdigit() or len(memory) > 2 or (len(memory) == 2 and memory[1] != 'MiB'):
                raise ValueError('invalid GPU process memory')
            pid = int(pid.strip())
            if current and pid == current[0]['pid']:
                observed[pid].append(int(memory[0]))
            else:
                other += 1
        other_counts.append(other)
    if not activity:
        raise ValueError('missing GPU telemetry')
    result_rows = {}
    for batch in batches:
        result_rows[str(batch)] = {}
        for runtime in ('rust', 'python'):
            selected = sorted((row for row in rows if row['batch'] == batch and row['runtime'] == runtime), key=lambda row: row['rep'])
            native_keys = (('used_peak', 'reserved_peak') if runtime == 'rust'
                           else ('torch_peak_allocated_bytes', 'torch_peak_reserved_bytes'))
            native = [row['native_allocator_bytes'] for row in selected]
            if any(not isinstance(entry, dict) or any(type(entry.get(key)) is not int or entry[key] <= 0 for key in native_keys) for entry in native):
                raise ValueError('missing native allocator counters')
            result_rows[str(batch)][runtime] = {
                **result['summary'][str(batch)][runtime],
                'process_medians_ms': [statistics.median(row['seconds']) * 1000 for row in selected],
                'load_seconds': [row['load_seconds'] for row in selected],
                'peak_rss_mib': [row['peak_rss_kib'] / 1024 for row in selected],
                'native_allocator_bytes': native,
                'sampled_gpu_peak_mib': [max(observed[row['pid']]) if observed[row['pid']] else None for row in selected],
                'gpu_samples_per_process': [len(observed[row['pid']]) for row in selected],
            }
    return {'size': size, 'head': manifest['head'], 'batches': result_rows,
            'activity_samples': len(activity), 'other_process_count_range': [min(other_counts), max(other_counts)],
            'native_scope': 'Rust CUDA default async pool vs Python PyTorch caching allocator; scopes differ',
            'sampled_gpu_scope': 'approximately 1 Hz driver process memory, not native peak'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    runs = []
    coverage = set()
    for directory in args.directories:
        files = {name: (directory / name).read_bytes() for name in ('manifest.json', 'result.json', 'activity.jsonl')}
        manifest = json.loads(files['manifest.json'])
        run = aggregate_run(manifest, json.loads(files['result.json']), [json.loads(line) for line in files['activity.jsonl'].splitlines()])
        for batch in run['batches']:
            identity = (run['size'], int(batch))
            if identity in coverage:
                raise ValueError('duplicate size/batch')
            coverage.add(identity)
        run['input_sha256'] = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        runs.append(run)
    if coverage != {(size, batch) for size, batches in EXPECTED.items() for batch in batches}:
        raise ValueError('incomplete size/batch coverage')
    print(json.dumps({'runs': runs}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
