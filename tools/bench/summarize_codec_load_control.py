"""Whole-process uncertainty for checked loads from full codec benchmark reports."""
import hashlib
import json
from pathlib import Path
import random
import statistics
from phase2_summary import codec, require, stats
from summarize_tokenizer_control import interval


def summarize(root, device):
    root=Path(root);environment=json.loads((root/'environment.json').read_bytes())
    count=environment['repeats']
    require(count>=3 and count%3==0 and 'codec' in environment['workloads'], 'incomplete codec control')
    if device=='cuda': require(environment['cpu_affinity'] is not None, 'missing CPU affinity')
    else:
        require(device=='metal' and environment.get('platform')=='Darwin'
                and environment.get('control_policy')=='counterbalanced_fresh_processes'
                and not environment.get('telemetry_error')
                and (root/'cpu-activity.jsonl').is_file(), 'missing Apple control telemetry')
    runs={};hashes={}
    for name in ('previous','candidate','python'):
        suffix='python' if name=='python' else 'rust'
        paths=[root/name/f'codec-{suffix}-{i}.json' for i in range(1,count+1)]
        require(set((root/name).glob(f'codec-{suffix}-*.json'))==set(paths), 'missing/extra codec processes')
        runs[name]=[json.loads(p.read_bytes()) for p in paths]
        hashes[name]=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        expected=('pytorch-pinned-eager-snapshot-fix' if device=='cuda' else 'python-mlx-adapted') if name=='python' else ('rust-candle' if device=='cuda' else 'rust-mlx')
        for run in runs[name]:
            require(run['runtime']==expected and run['schema_version']==1 and run['warmups']==5 and run['trials']==30, 'codec protocol mismatch')
            require(not run.get('diagnostic_profiling',False) and run['source_sha256']==runs[name][0]['source_sha256'], 'instrumented/changing codec')
            require(len(run['load_trials_ms'])==30, 'wrong load trial count')
    # Validate every process's complete outputs/scopes in original three-process
    # groups. No timing samples are duplicated or discarded for the bootstrap.
    for name in ('previous','candidate'):
        for start in range(0,count,3):
            codec(root,device,runs={'rust':runs[name][start:start+3], 'python':runs['python'][start:start+3]})
    trials={name:[r['load_trials_ms'] for r in group] for name,group in runs.items()}
    summaries={name:stats([v for r in group for v in r]) | {'process_medians_ms':[statistics.median(r) for r in group]} for name,group in trials.items()}
    comparisons={name:{'median_ratio':summaries['candidate']['median_ms']/summaries[name]['median_ms'],
                       'p95_ratio':summaries['candidate']['p95_ms']/summaries[name]['p95_ms'],
                       'paired_process_percentile_95_interval':interval(trials['candidate'],trials[name],random.Random(20261110))} for name in ('previous','python')}
    return {'schema_version':1,'processes_per_runtime':count,'cpu_affinity':environment['cpu_affinity'],
            'scope':'Full codec reports; only checked-load uncertainty is adjudicated here.',
            'report_sha256':hashes,'measurements':[{'scope':'checked_load','timings':summaries,'candidate_over':comparisons}]}
