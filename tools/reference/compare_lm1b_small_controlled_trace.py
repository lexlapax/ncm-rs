"""Compare experimental LM1B-small controlled stochastic output against the pinned CUDA trace."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from codec_trace import POLICY, category
from tinystories_artifact import digest_file, json_file

REFERENCE = '173f1be79510d9fabad465652e748a43d5d71d447c20acaa581e4c1476689c64'


def compare_tensor(name, expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(f'shape or dtype mismatch: {name}')
    if expected.is_floating_point():
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError(f'nonfinite: {name}')
        policy = category(name)
        rule = POLICY[policy]
        errors = (expected.double() - actual.double()).abs()
        result = {'name': name, 'elements': expected.numel(), 'policy': policy,
                  'max_abs_error': errors.max().item(),
                  'mismatches': (errors > rule['atol'] + rule['rtol'] * expected.double().abs()).sum().item()}
        if name.startswith('generator.level_') and name.endswith('.logits'):
            result['argmax_mismatches'] = (expected.argmax(-1) != actual.argmax(-1)).sum().item()
            result['argmax_choices'] = expected.numel() // expected.shape[-1]
        return result
    return {'name': name, 'elements': expected.numel(), 'dtype': str(expected.dtype),
            'mismatches': (expected != actual).sum().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() or digest_file(args.reference) != REFERENCE:
        raise ValueError('output exists or reference hash differs')
    candidate_sha = digest_file(args.candidate)
    with safe_open(args.reference, framework='pt', device='cpu') as ref, safe_open(args.candidate, framework='pt', device='cpu') as candidate:
        if ref.keys() != candidate.keys():
            raise ValueError(f'trace inventory differs: missing={sorted(set(ref.keys()) - set(candidate.keys()))}, extra={sorted(set(candidate.keys()) - set(ref.keys()))}')
        rows = [compare_tensor(name, ref.get_tensor(name), candidate.get_tensor(name)) for name in ref.keys()]
    failures = [row for row in rows if row['mismatches'] or row.get('argmax_mismatches', 0)]
    result = {'schema_version': 1, 'scope': 'two-row controlled stochastic generator comparison; not stochastic quality or performance',
              'reference_sha256': REFERENCE, 'candidate_sha256': candidate_sha, 'policy': POLICY,
              'trace_count': len(rows), 'all_passed': not failures, 'failures': failures, 'traces': rows}
    json_file(args.output, result)
    print(f'{len(rows)} tensors; {len(failures)} fail the unchanged policy')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
