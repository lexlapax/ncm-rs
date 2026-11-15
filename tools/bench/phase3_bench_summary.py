"""Validate and summarize the three-process Phase 3 CUDA timing capture."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def percentile(xs, q):
    return sorted(xs)[math.ceil(q * len(xs)) - 1]

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('capture', type=Path); p.add_argument('--source-root', type=Path, default=ROOT); a = p.parse_args()
    manifest = json.loads((a.capture/'manifest.json').read_bytes())
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((a.source_root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'source or binary changed: {name}')
    if manifest['batches'] != [1, 4, 32, 128] or manifest['warmup_per_process'] != 5 or manifest['trials_per_process'] != 30 or manifest['processes_per_runtime_batch'] != 3:
        raise ValueError('invalid measurement plan')
    result = {}
    for batch in manifest['batches']:
        candidate = {}; hashes = set()
        for runtime in ('rust', 'python'):
            samples, process_medians, rss = [], [], []
            device_peaks = []
            cuda_pools = []
            for rep in range(3):
                stem=f'{runtime}-b{batch}-r{rep}'
                report=json.loads((a.capture/(stem+'.json')).read_bytes())
                if report['batch'] != batch or report['warmups'] != 5 or len(report['trials']) != 30:
                    raise ValueError(f'incomplete {stem}')
                values=report['trials']
                if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values): raise ValueError('bad timing sample')
                hashes.add(report['outputs_sha256_i64le']); samples.extend(values); process_medians.append(statistics.median(values))
                timing=(a.capture/(stem+'.time')).read_text()
                if 'Exit status: 0' not in timing: raise ValueError(f'nonzero process {stem}')
                memory=[x.split(':', 1)[1].strip() for x in timing.splitlines() if 'Maximum resident set size (kbytes):' in x]
                if len(memory) != 1: raise ValueError('missing host peak RSS')
                rss.append(int(memory[0]) * 1024)
                if runtime == 'python': device_peaks.append({'allocated': report['torch_allocated_peak'], 'reserved': report['torch_reserved_peak']})
                if runtime == 'rust' and report.get('cuda_pool') is not None: cuda_pools.append(report['cuda_pool'])
            candidate[runtime]={'sample_count':len(samples), 'median_ms':statistics.median(samples)*1000,
                                'p95_ms_nearest_rank':percentile(samples,.95)*1000,
                                'process_medians_ms':[x*1000 for x in process_medians],
                                'min_ms':min(samples)*1000, 'max_ms':max(samples)*1000,
                                'peak_rss_bytes':rss, 'torch_device_peaks':device_peaks or None}
            if cuda_pools:
                if len(cuda_pools) != 3: raise ValueError('missing CUDA pool observation')
                candidate[runtime]['cuda_default_pool'] = cuda_pools
        if len(hashes) != 1: raise ValueError(f'output mismatch for batch {batch}')
        candidate['rust_over_python_median']=candidate['rust']['median_ms']/candidate['python']['median_ms']
        candidate['outputs_sha256_i64le']=hashes.pop();result[str(batch)]=candidate
    telemetry=[json.loads(line) for line in (a.capture/'activity.jsonl').read_text().splitlines()]
    if not telemetry or any(len(x['queries'])!=2 or any(q.get('status')!=0 for q in x['queries']) for x in telemetry):
        raise ValueError('activity observation incomplete')
    events_path = a.capture / 'process-events.jsonl'
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        for batch in manifest['batches']:
            for runtime in ('rust', 'python'):
                observed = []
                for rep in range(3):
                    stem = f'{runtime}-b{batch}-r{rep}'
                    starts = [x for x in events if x['name'] == stem and x['event'] == 'start']
                    ends = [x for x in events if x['name'] == stem and x['event'] == 'end']
                    if len(starts) != 1 or len(ends) != 1 or ends[0]['returncode'] != 0: raise ValueError('invalid process event pair')
                    samples = []
                    for sample in telemetry:
                        if not starts[0]['monotonic'] <= sample['monotonic'] <= ends[0]['monotonic']: continue
                        for row in sample['queries'][1]['stdout'].splitlines():
                            parts = row.split(',')
                            try:
                                pid = int(parts[0].strip())
                                mib = int(parts[-1].strip().split()[0])
                            except (ValueError, IndexError): continue
                            if sample.get('parents', {}).get(str(pid)) == starts[0]['pid']: samples.append(mib)
                    if not samples: raise ValueError(f'no process GPU memory samples for {stem}')
                    observed.append({'peak_mib':max(samples), 'sample_count':len(samples)})
                result[str(batch)][runtime]['gpu_process_sampled_memory'] = observed
    has_pools = all('cuda_default_pool' in result[str(b)]['rust'] for b in manifest['batches'])
    memory_limit = ('Allocator counters have different coverage across Rust and PyTorch.' if has_pools
                    else 'Rust native CUDA-pool peak not captured in this timing run.')
    print(json.dumps({'scope':manifest['scope'], 'batches':result, 'activity_samples':len(telemetry),
                      'activity_warning':'GPU process observations are sampled, not continuous; inspect activity.jsonl for competing work.',
                      'limitations':[memory_limit,
                                     'Host peak RSS is whole-process, including model load and runtime startup.',
                                     'Timings are a single machine, not universal performance claims.']},indent=2,sort_keys=True))

if __name__ == '__main__': main()
