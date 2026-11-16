"""Inspect CUDA penalized-logit margins at Metal's observed differing positions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from xsum_artifact import digest_file
from xsum_numpy_selection import select

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / 'reports/phase4-xsum-top1-divergence-cuda-v1/trace.safetensors'
TRACE_SHA = '42dc76b4ed384a76f3dc375890d86c0e33f8ad74a1e2b0753ba3c3f6524d1895'
INFERENCE = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final/inference.safetensors'
MANIFEST = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final/manifest.json'
MANIFEST_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
METAL = ROOT / 'reports/phase4-xsum-mlx-full-top1-captured-v1'
ROWS = (4883, 4937)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if (digest_file(TRACE) != TRACE_SHA or digest_file(MANIFEST) != MANIFEST_SHA
            or digest_file(METAL / 'manifest.json') != 'b0e6efa956268e3e424ac5a37a11f60b860bcedf8b2c1ac7661a2f4dc4d91923'):
        raise ValueError('reference trace or manifest differs')
    manifest = json.loads(MANIFEST.read_bytes())
    if digest_file(INFERENCE) != manifest['files']['inference.safetensors']['sha256']:
        raise ValueError('inference payload differs')
    trace = load_file(TRACE)
    logits = trace['generator.final_logits'].numpy()
    expected = trace['generator.penalized_ids'].numpy()
    unigram = load_file(INFERENCE)['token_unigram'].numpy()
    allowance = unigram.astype(np.float64) * 48
    if logits.shape != (2, 48, 50304) or not np.array_equal(select(logits, unigram), expected):
        raise ValueError('NumPy selector differs from pinned CUDA trace')
    result = []
    for batch, index in enumerate(ROWS):
        path = METAL / f'batch-{index // 32 * 32:05}.json'
        envelope = json.loads(path.read_bytes())
        raw = json.dumps(envelope['record'], sort_keys=True, separators=(',', ':')).encode()
        if (envelope.get('sha256') != hashlib.sha256(raw).hexdigest()
                or envelope['record'].get('start') != index // 32 * 32
                or envelope['record'].get('count') != 32):
            raise ValueError('Metal batch receipt differs')
        metal = envelope['record']['result']['ids'][index % 32]
        counts = np.zeros(50304, dtype=np.float64)
        history = []
        for t in range(48):
            score = logits[batch, t].astype(np.float64)
            score -= 2.0 * np.maximum(counts - allowance, 0.0)
            score += ((48 - t) / 48) * np.maximum(allowance - counts, 0.0)
            for j, penalty in enumerate((5.0, 2.5, 1.25, 0.625)):
                if j < t:
                    score[history[-j - 1]] -= penalty
            choice = int(np.argmax(score))
            if choice != int(expected[batch, t]):
                raise ValueError('selector history differs')
            if metal[t] != choice:
                top = np.partition(score, -2)[-2:]
                result.append({'row': index, 'position': t, 'cuda_id': choice, 'metal_id': metal[t],
                               'cuda_top_two_gap': float(top[1] - top[0]),
                               'cuda_winner_minus_metal_id_score': float(score[choice] - score[metal[t]]),
                               'prior_raw_id_differences': int(sum(a != b for a, b in zip(metal[:t], expected[batch, :t])))})
            history.append(choice)
            counts[choice] += 1.0
    result = {'trace_sha256': TRACE_SHA, 'rows': list(ROWS), 'differences': result,
              'scope': 'CUDA fixed Python code history and CUDA prior token choices only; not Metal logits or a cause'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
