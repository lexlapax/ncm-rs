"""Audit the controlled LM1B divergence without changing either capture."""
import argparse
import json
from pathlib import Path

from safetensors import safe_open
from lm1b_small_controlled_trace import ControlledRandom
from tinystories_artifact import ROOT, digest_file

REFERENCE = '173f1be79510d9fabad465652e748a43d5d71d447c20acaa581e4c1476689c64'
CANDIDATE = '8d5184bc9ffeb0600c724d2454682cf3acf79e6488de4079fd8d4c162bb364e5'
LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if (args.output.exists() or args.output.is_symlink() or
            not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()) or
            digest_file(args.reference) != REFERENCE or digest_file(args.candidate) != CANDIDATE):
        raise ValueError('output or input identity differs')
    inference = ROOT / 'artifacts/canonical/lm1b-small-inference-v1/inference.safetensors'
    if digest_file(inference.parent / 'manifest.json') != '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76':
        raise ValueError('artifact manifest differs')
    manifest = json.loads((inference.parent / 'manifest.json').read_bytes())
    record = manifest['files']['inference.safetensors']
    if digest_file(inference) != record['sha256']:
        raise ValueError('inference distribution differs')
    rng = ControlledRandom(11)
    with safe_open(inference, framework='pt', device='cpu') as f:
        distribution = f.get_tensor('level0_unigram').tolist()
    with safe_open(args.reference, framework='pt', device='cpu') as ref, safe_open(args.candidate, framework='pt', device='cpu') as cand:
        reference, codes = ref.get_tensor('generator.generated_codes'), cand.get_tensor('generator.generated_codes')
        mismatches = sum(rng.choose(distribution) != int(codes[row, 0]) for row in range(2))
        for level, length in enumerate(LEVELS[1:], 1):
            offset = sum(LEVELS[:level])
            rows = cand.get_tensor(f'generator.level_{level:02}.logits').reshape(-1, 8192).tolist()
            mismatches += sum(rng.logits(values) != int(codes[i // length, offset + i % length]) for i, values in enumerate(rows))
        first = (reference != codes).nonzero()[0].tolist()
        if first != [1, sum(LEVELS[:7]) + 43]:
            raise ValueError('unexpected first difference')
        rng = ControlledRandom(11)
        draw_index = 2 + 2 * sum(LEVELS[1:7]) + 64 + 43
        for _ in range(draw_index):
            rng.uniform()
        uniform = rng.uniform()
        boundaries = []
        import math
        for f in (ref, cand):
            values = f.get_tensor('generator.level_07.logits')[1, 43].tolist()
            maximum = max(values)
            weights = [math.exp(x - maximum) for x in values]
            boundaries.append(sum(weights[:3213]) / sum(weights))
    result = {'scope': 'one observed near-boundary divergence; same-logit sampler replay',
              'reference_sha256': REFERENCE, 'candidate_sha256': CANDIDATE,
              'same_logit_choices': 1282, 'same_logit_mismatches': mismatches,
              'first_difference': first, 'draw_index': draw_index, 'uniform': uniform,
              'reference_boundary': boundaries[0], 'candidate_boundary': boundaries[1]}
    if mismatches or not boundaries[0] < uniform < boundaries[1]:
        raise ValueError('unexpected same-logit replay or boundary')
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
