"""Summarize complete alternating TinyStories CUDA benchmark captures."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest['batches'] != [1, 4, 32, 128] or manifest['processes_per_runtime_batch'] != 3:
        raise ValueError('expected all batches and three processes per runtime')
    events = [json.loads(line) for line in (args.run / 'process-events.jsonl').read_text().splitlines()]
    samples = [json.loads(line) for line in (args.run / 'activity.jsonl').read_text().splitlines()]
    results = {}
    for batch in manifest['batches']:
        per_runtime = {}
        for runtime in ('python', 'rust'):
            medians, timings, rss, device_peaks, cleanup_medians = [], [], [], [], []
            for rep in range(3):
                stem = f'{runtime}-b{batch}-r{rep}'
                starts = [x for x in events if x['name'] == stem and x['event'] == 'start']
                ends = [x for x in events if x['name'] == stem and x['event'] == 'end']
                if len(starts) != 1 or len(ends) != 1 or ends[0]['returncode'] != 0:
                    raise ValueError(f'incomplete process: {stem}')
                data = json.loads((args.run / (stem + '.json')).read_text())
                vals = data['trials_seconds']
                if data['batch'] != batch or data['warmups'] != 5 or len(vals) != 30 or any(not math.isfinite(v) or v <= 0 for v in vals):
                    raise ValueError(f'invalid timing: {stem}')
                medians.append(statistics.median(vals) * 1000)
                timings.extend(v * 1000 for v in vals)
                if runtime == 'python':
                    cleanup = data.get('cleanup_seconds', [])
                    if len(cleanup) != 30 or any(not math.isfinite(v) or v < 0 for v in cleanup):
                        raise ValueError(f'invalid cleanup timing: {stem}')
                    cleanup_medians.append(statistics.median(cleanup) * 1000)
                text = (args.run / (stem + '.time')).read_text()
                m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', text)
                if not m or not re.search(r'Exit status:\s*0\b', text):
                    raise ValueError(f'invalid RSS capture: {stem}')
                rss.append(int(m.group(1)) / 1024)
                peaks = []
                for sample in samples:
                    if not starts[0]['monotonic'] <= sample['monotonic'] <= ends[0]['monotonic']:
                        continue
                    owners = sample.get('parents', {})
                    for row in sample['queries'][1].get('stdout', '').splitlines():
                        try:
                            pid = int(row.split(',', 1)[0].strip())
                            mib = int(row.rsplit(',', 1)[1].strip().split()[0])
                        except (ValueError, IndexError):
                            continue
                        if pid == starts[0]['pid'] or owners.get(str(pid)) == starts[0]['pid']:
                            peaks.append(mib)
                device_peaks.append(max(peaks) if peaks else None)
            per_runtime[runtime] = {'median_process_median_ms': statistics.median(medians),
                                    'process_medians_ms': medians, 'p95_pooled_ms': percentile(timings, .95),
                                    'median_peak_rss_mib': statistics.median(rss), 'peak_rss_mib': rss,
                                    'sampled_gpu_peak_mib': device_peaks,
                                    'python_cleanup_medians_ms': cleanup_medians}
        py, rust = per_runtime['python'], per_runtime['rust']
        results[str(batch)] = {'python': py, 'rust': rust,
                               'rust_over_python_latency': rust['median_process_median_ms'] / py['median_process_median_ms'],
                               'rust_over_python_peak_rss': rust['median_peak_rss_mib'] / py['median_peak_rss_mib']}
    print(json.dumps({'batches': results, 'gpu_memory_caveat': '1 Hz process samples are lower bounds and may miss transient peaks',
                      'latency_scope': manifest['scope'], 'sampling': manifest['sampling'],
                      'python_cleanup': manifest['python_cleanup']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
