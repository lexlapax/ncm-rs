"""Exploratory bootstrap decomposition; does not alter the frozen quality gate."""
import json
import numpy as np
from pathlib import Path

seeds = [42, 0, 1, 2, 3]
runtimes = ('python', 'rust')
keys = ('rouge1', 'rouge2', 'rougeL', 'generation_perplexity', 'mean_words')
a = {}
for runtime in runtimes:
    records = [json.loads(Path(f'reports/phase4-xsum-stochastic-qualification-heldout-{runtime}-{s}-score-v1.json').read_bytes()) for s in seeds]
    a[runtime] = {}
    for k in ('rouge1', 'rouge2', 'rougeL'):
        a[runtime][k] = np.array([[100 * r['rouge'][k] for r in x['rows']] for x in records])
    for k in ('nll', 'tokens', 'words'):
        a[runtime][k] = np.array([[r[k] for r in x['rows']] for x in records])

def scores(v, seeds_, rows):
    ret = {k: v[k][seeds_][:, rows].mean() for k in keys[:3]}
    ret['mean_words'] = v['words'][seeds_][:, rows].mean()
    nll, count = (v[k][seeds_][:, rows].sum(axis=1) for k in ('nll', 'tokens'))
    ret['generation_perplexity'] = np.exp(nll / count).mean()
    return ret

def diff(p, r):
    return {k: float(r[k] - p[k]) for k in keys}

rng = np.random.default_rng(20261117)
outputs = {}
for mode in ('seed_only', 'row_only', 'both'):
    samples = {k: [] for k in keys}
    for _ in range(10000):
        rows = rng.integers(0, 256, 256) if mode != 'seed_only' else np.arange(256)
        ps = rng.integers(0, 5, 5) if mode != 'row_only' else np.arange(5)
        rs = rng.integers(0, 5, 5) if mode != 'row_only' else np.arange(5)
        delta = diff(scores(a['python'], ps, rows), scores(a['rust'], rs, rows))
        for k in keys: samples[k].append(delta[k])
    outputs[mode] = {k: {'sd': float(np.std(v, ddof=1)), 'p025': float(np.quantile(v, .025)),
                         'p975': float(np.quantile(v, .975))} for k, v in samples.items()}
print(json.dumps({'bootstrap_seed':20261117,'replicates_per_mode':10000,'difference':'rust minus python',
                  'status':'post-hoc diagnostic; frozen gate unchanged','modes':outputs}, indent=2))
