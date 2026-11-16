"""Apply the frozen XSum follow-up bootstrap to Python and Metal score files."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from xsum_artifact import digest_file
from xsum_mlx_stochastic_score import DATA_SHA, PROTOCOL_SHA, REFERENCE_SHA, ROOT
from xsum_stochastic_followup_analyze import metrics
from xsum_stochastic_followup_capture import METRICS, validate_protocol

FOLLOWUP_RESULT_SHA = '4c788e0509a40523a4e1a3b789843f817ee16465470014598150b4a3cb9c79eb'


def bootstrap(python, metal, replicates=10000, seed=20261116):
    def matrix(items, key):
        if key in ('rouge1', 'rouge2', 'rougeL'):
            return np.array([[100 * r['rouge'][key] for r in item['rows']] for item in items])
        return np.array([[r[key] for r in item['rows']] for item in items])
    arrays = [{k: matrix(items, k) for k in ('rouge1', 'rouge2', 'rougeL', 'words', 'nll', 'tokens')}
              for items in (python, metal)]
    def summarize(a, seeds, rows):
        result = {k: a[k][seeds][:, rows].mean() for k in ('rouge1', 'rouge2', 'rougeL')}
        result['mean_words'] = a['words'][seeds][:, rows].mean()
        nll = a['nll'][seeds][:, rows].sum(axis=1)
        tokens = a['tokens'][seeds][:, rows].sum(axis=1)
        if np.any(tokens <= 0):
            raise ValueError('bootstrap sample has no scoreable tokens')
        result['generation_perplexity'] = np.exp(nll / tokens).mean()
        return result
    def degradation(p, m):
        return {k: (abs(m[k] - p[k]) if k == 'mean_words' else m[k] - p[k]
                    if k == 'generation_perplexity' else p[k] - m[k]) for k in METRICS}
    size = len(python[0]['rows'])
    if len(python) != 5 or len(metal) != 5 or any(len(item['rows']) != size for item in python + metal):
        raise ValueError('five equal-sized seed sets required')
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in METRICS}
    for _ in range(replicates):
        rows = rng.integers(0, size, size)
        p = summarize(arrays[0], rng.integers(0, 5, 5), rows)
        m = summarize(arrays[1], rng.integers(0, 5, 5), rows)
        for k, value in degradation(p, m).items():
            boot[k].append(float(value))
    observed = degradation(summarize(arrays[0], np.arange(5), np.arange(size)),
                           summarize(arrays[1], np.arange(5), np.arange(size)))
    return observed, {k: float(np.quantile(values, 0.975)) for k, values in boot.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal-scores', nargs=5, type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    protocol_path = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-followup-protocol-v1.json'
    prior_path = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-followup-result-v1.json'
    if digest_file(protocol_path) != PROTOCOL_SHA or digest_file(prior_path) != FOLLOWUP_RESULT_SHA:
        raise ValueError('frozen protocol or published result differs')
    protocol = json.loads(protocol_path.read_bytes()); validate_protocol(protocol)
    published = json.loads(prior_path.read_bytes())
    if published['protocol_sha256'] != PROTOCOL_SHA or published['python_score_sha256'][0] != REFERENCE_SHA:
        raise ValueError('unexpected retained Python scores')
    replay = json.loads(args.replay.read_bytes())
    if (replay.get('runtime') != 'replay' or replay.get('replay_exact_rows') is not True
            or replay.get('replay_exact_metrics') is not True or replay.get('model_sha256') is None
            or REFERENCE_SHA not in replay.get('capture_files_sha256', {}).values()):
        raise ValueError('exact scorer replay not established')
    python, metal = [], []
    for i, seed in enumerate((4, 5, 6, 7, 8)):
        py_path = ROOT / f'reports/phase4-xsum-stochastic-followup-python-{seed}-score-v1.json'
        if digest_file(py_path) != published['python_score_sha256'][i]:
            raise ValueError('retained Python score differs')
        py = json.loads(py_path.read_bytes())
        mx = json.loads(args.metal_scores[i].read_bytes())
        if (py['manifest'].get('seed') != seed or py['manifest'].get('data_sha256') != DATA_SHA
                or py.get('scorer_sha256') != protocol.get('scorer_sha256')
                or any(py.get(k) != replay.get(k) for k in ('genppl_sha256', 'model_sha256', 'versions', 'torch'))
                or mx.get('runtime') != 'metal' or mx['manifest'].get('seed') != seed
                or mx['manifest'].get('protocol_sha256') != PROTOCOL_SHA
                or any(mx.get(k) != replay.get(k) for k in ('scorer_sha256', 'scorer_dependencies_sha256', 'genppl_sha256', 'model_sha256', 'versions', 'torch', 'gpu_identity_and_driver'))):
            raise ValueError('score or replay provenance differs')
        for obj in (py, mx):
            if [row['index'] for row in obj['rows']] != list(range(512, 2560)):
                raise ValueError('score rows differ')
            calculated = metrics(obj['rows'])
            if set(obj['metrics']) != set(METRICS) or any(
                not math.isclose(calculated[k], obj['metrics'][k], rel_tol=1e-12, abs_tol=1e-12) for k in METRICS
            ):
                raise ValueError('score metrics differ from rows')
        python.append(py); metal.append(mx)
    observed, bounds = bootstrap(python, metal)
    result = {'schema_version': 1, 'protocol_sha256': PROTOCOL_SHA, 'replay_sha256': digest_file(args.replay),
              'python_score_sha256': published['python_score_sha256'],
              'metal_score_sha256': [digest_file(p) for p in args.metal_scores],
              'observed_degradation': observed, 'bootstrap_upper_97_5': bounds,
              'margins': protocol['margins'], 'passes': {k: bounds[k] <= protocol['margins'][k] for k in METRICS},
              'scope': 'additional Metal backend comparison using previously inspected Python follow-up outputs'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True); out.write('\n')
    if not all(result['passes'].values()):
        raise SystemExit('Metal stochastic quality bounds failed; comparison retained')


if __name__ == '__main__':
    main()
