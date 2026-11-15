"""Estimate sequential process windows in a recorded MLX comparison's CPU activity."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    manifest, manifest_hash = load(args.run / 'manifest.json')
    activity_bytes = (args.run / 'activity.jsonl').read_bytes()
    activity = [json.loads(line) for line in activity_bytes.splitlines()]
    if not manifest.get('stable_inputs_and_sources') or manifest.get('telemetry_errors'):
        raise ValueError('benchmark reports changed sources or telemetry errors')
    if not activity:
        raise ValueError('no activity samples')
    times = [row['monotonic'] for row in activity]
    if any(not math.isfinite(t) for t in times) or times != sorted(times):
        raise ValueError('invalid sample timestamps')
    origin = times[0]
    elapsed = 0.0
    windows = []
    for entry in manifest['runs']:
        name = entry['name']
        if not name or '/' in name or name in ('.', '..'):
            raise ValueError('invalid process name')
        meta, meta_hash = load(args.run / f'{name}.meta.json')
        duration = meta['wall_seconds_including_load']
        if meta['returncode'] != 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'invalid process duration: {name}')
        start, end = elapsed, elapsed + duration
        samples = [row for row in activity if start <= row['monotonic'] - origin < end]
        background = {}
        for row in samples:
            for proc in row['top_cpu']:
                name_key = proc['name']
                if name_key in ('python', 'tinystories_lifecycle_bench', '(tinystories_life)'):
                    continue
                item = background.setdefault(name_key, {'samples': 0, 'max_cpu_percent': 0.0})
                item['samples'] += 1
                item['max_cpu_percent'] = max(item['max_cpu_percent'], proc['cpu_percent'])
        windows.append({'name': name, 'estimated_start_seconds': start, 'estimated_end_seconds': end,
                        'metadata_sha256': meta_hash, 'sample_count': len(samples), 'background': background})
        elapsed = end
    report = {'manifest_sha256': manifest_hash, 'activity_sha256': hashlib.sha256(activity_bytes).hexdigest(),
              'activity_samples': len(activity), 'activity_span_seconds': times[-1] - origin,
              'sum_process_wall_seconds': elapsed, 'windows': windows,
              'limitations': ['Process windows are estimated from the first sample and sequential child durations;',
                              'launcher gaps and monitoring overhead are not measured and can shift boundaries.',
                              'CPU snapshots are sampled; GPU activity and exact trial timing are unavailable.',
                              'Temporal overlap does not establish causation.']}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'estimated {len(windows)} process windows from {len(activity)} CPU samples')


if __name__ == '__main__':
    main()
