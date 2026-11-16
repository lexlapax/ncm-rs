"""Freeze development-derived XSum margins or analyze held-out stochastic scores."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from xsum_artifact import digest_file
from xsum_stochastic_followup_capture import DATA_SHA, METRICS, ROOT, SEEDS, validate_protocol


def metrics(rows):
    count = sum(r['tokens'] for r in rows)
    if count <= 0:
        raise ValueError('no scoreable tokens')
    out = {k: 100 * sum(r['rouge'][k] for r in rows) / len(rows) for k in ('rouge1', 'rouge2', 'rougeL')}
    out['mean_words'] = sum(r['words'] for r in rows) / len(rows)
    out['generation_perplexity'] = math.exp(sum(r['nll'] for r in rows) / count)
    return out


def load(paths, runtime, subset, protocol_sha=None):
    if len(paths) != 5:
        raise ValueError('five score files required')
    items = []
    for path, seed in zip(paths, SEEDS[subset]):
        obj = json.loads(path.read_bytes())
        manifest = obj['manifest']
        start = 0 if subset == 'development' else 512
        if (obj.get('schema_version') != 1 or manifest.get('runtime') != runtime
                or manifest.get('set') != subset or manifest.get('seed') != seed
                or manifest.get('data_sha256') != DATA_SHA or manifest.get('protocol_sha256') != protocol_sha
                or [r['index'] for r in obj['rows']] != list(range(start, start + 2048))
                or set(obj['metrics']) != set(METRICS)):
            raise ValueError(f'invalid score file: {path}')
        calculated = metrics(obj['rows'])
        if any(not math.isclose(calculated[k], obj['metrics'][k], rel_tol=1e-12, abs_tol=1e-12) for k in METRICS):
            raise ValueError(f'metric mismatch: {path}')
        items.append(obj)
    for obj in items[1:]:
        if any(obj[k] != items[0][k] for k in ('scorer_sha256', 'genppl_sha256', 'model_sha256', 'versions', 'torch')):
            raise ValueError('scorer identities differ')
        if any(obj['manifest'][k] != items[0]['manifest'][k] for k in ('source_sha256', 'native_libraries', 'gpu_identity_and_driver')):
            raise ValueError('capture identities differ')
    return items


def compare(python_paths, rust_paths, protocol_path):
    protocol = json.loads(protocol_path.read_bytes()); validate_protocol(protocol)
    sha = digest_file(protocol_path)
    py = load(python_paths, 'python', 'heldout', sha)
    rs = load(rust_paths, 'rust', 'heldout', sha)
    if py[0]['scorer_sha256'] != protocol['scorer_sha256'] or rs[0]['scorer_sha256'] != protocol['scorer_sha256']:
        raise ValueError('scorer changed since freeze')
    for key in ('source_sha256', 'native_libraries', 'gpu_identity_and_driver'):
        if py[0]['manifest'][key] != rs[0]['manifest'][key]:
            raise ValueError('runtime capture environments differ')
    def matrix(items, key):
        if key in ('rouge1', 'rouge2', 'rougeL'):
            return np.array([[100 * r['rouge'][key] for r in item['rows']] for item in items])
        return np.array([[r[key] for r in item['rows']] for item in items])
    arrays = [{k: matrix(items, k) for k in ('rouge1', 'rouge2', 'rougeL', 'words', 'nll', 'tokens')} for items in (py, rs)]
    def summarize(a, seeds, rows):
        result = {k: a[k][seeds][:, rows].mean() for k in ('rouge1', 'rouge2', 'rougeL')}
        result['mean_words'] = a['words'][seeds][:, rows].mean()
        nll = a['nll'][seeds][:, rows].sum(axis=1)
        tokens = a['tokens'][seeds][:, rows].sum(axis=1)
        if np.any(tokens <= 0): raise ValueError('bootstrap sample has no scoreable tokens')
        result['generation_perplexity'] = np.exp(nll / tokens).mean()
        return result
    def degradation(p, r):
        return {k: (abs(r[k] - p[k]) if k == 'mean_words' else r[k] - p[k]
                    if k == 'generation_perplexity' else p[k] - r[k]) for k in METRICS}
    rng = np.random.default_rng(20261116)
    boot = {k: [] for k in METRICS}
    for _ in range(10000):
        rows = rng.integers(0, 2048, 2048)
        p = summarize(arrays[0], rng.integers(0, 5, 5), rows)
        r = summarize(arrays[1], rng.integers(0, 5, 5), rows)
        for k, v in degradation(p, r).items(): boot[k].append(float(v))
    bounds = {k: float(np.quantile(v, 0.975)) for k, v in boot.items()}
    observed = degradation(summarize(arrays[0], np.arange(5), np.arange(2048)),
                           summarize(arrays[1], np.arange(5), np.arange(2048)))
    return {'schema_version': 1, 'protocol_sha256': sha,
            'python_score_sha256': [digest_file(p) for p in python_paths],
            'rust_score_sha256': [digest_file(p) for p in rust_paths],
            'python_seed_metrics': [x['metrics'] for x in py], 'rust_seed_metrics': [x['metrics'] for x in rs],
            'observed_degradation': observed, 'bootstrap_upper_97_5': bounds, 'margins': protocol['margins'],
            'passes': {k: bounds[k] <= protocol['margins'][k] for k in METRICS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-scores', nargs=5, type=Path, required=True)
    parser.add_argument('--rust-scores', nargs=5, type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('new output within repository required')
    result = compare(args.python_scores, args.rust_scores, args.protocol)
    with args.output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
