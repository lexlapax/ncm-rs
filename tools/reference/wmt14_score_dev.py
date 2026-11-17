"""Score WMT14 development text with the pinned upstream metrics and bootstrap."""
import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

VERSIONS = {'colorama': '0.4.6', 'lxml': '6.1.3', 'numpy': '2.5.5', 'portalocker': '4.4.2',
            'regex': '2026.11.15', 'sacrebleu': '2.5.1', 'tabulate': '0.10.0'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(hypotheses, references):
    if not hypotheses or len(hypotheses) != len(references) or any(type(s) is not str for s in hypotheses + references):
        raise ValueError('invalid hypotheses or references')
    metrics = {'bleu': BLEU(), 'bleu_lowercase': BLEU(lowercase=True), 'chrf': CHRF()}
    values, signatures = {}, {}
    for name, metric in metrics.items():
        values[name] = metric.corpus_score(hypotheses, [references]).score
        signatures[name] = str(metric.get_signature())
    denominator = sum(len(x.split()) for x in references)
    if not denominator:
        raise ValueError('zero reference length')
    values['length_ratio'] = sum(len(x.split()) for x in hypotheses) / denominator
    return values, signatures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--input-sha256', required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or digest(args.input) != args.input_sha256:
        raise ValueError('output exists or input hash differs')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'scorer dependency differs: {name}')
    data = json.loads(args.input.read_bytes())
    if data['start'] != 0 or data['count'] != 128 or len(data['hypotheses']) != 128 or len(data['references']) != 128:
        raise ValueError('only the designated development set is supported')
    hypotheses, references = data['hypotheses'], data['references']
    values, signatures = score(hypotheses, references)
    rng = random.Random(42)
    samples = {key: [] for key in values}
    for _ in range(1000):
        indices = [rng.randrange(128) for _ in range(128)]
        metrics, _ = score([hypotheses[i] for i in indices], [references[i] for i in indices])
        for key, value in metrics.items():
            samples[key].append(value)
    intervals = {key: [sorted(values)[24], sorted(values)[974]] for key, values in samples.items()}
    if digest(args.input) != args.input_sha256:
        raise ValueError('input changed during scoring')
    result = {'scope': '128-row development scoring and baseline bootstrap, not held-out quality',
              'input_sha256': args.input_sha256, 'versions': VERSIONS, 'metrics': values,
              'signatures': signatures, 'bootstrap': {'seed': 42, 'resamples': 1000, 'intervals_95': intervals}}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
