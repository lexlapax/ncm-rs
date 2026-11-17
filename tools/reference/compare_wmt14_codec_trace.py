"""Compare an experimental CUDA codec capture against the frozen WMT14 trace."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from codec_trace import POLICY, category
from tinystories_artifact import digest_file, json_file

REFERENCE = 'b4b66ab0b58910e501aed704d50df9493bad65d691e78738f8689d14649662fd'


def compare_tensor(name, expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(f'shape or dtype mismatch: {name}')
    if expected.is_floating_point():
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError(f'nonfinite: {name}')
        policy = category(name)
        rule = POLICY[policy]
        errors = (expected.double() - actual.double()).abs()
        return {'name': name, 'elements': expected.numel(), 'policy': policy,
                'max_abs_error': errors.max().item(),
                'mismatches': (errors > rule['atol'] + rule['rtol'] * expected.double().abs()).sum().item()}
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
    failures = [row for row in rows if row['mismatches']]
    result = {'schema_version': 1, 'scope': 'two-row experimental codec comparison; not quality or performance',
              'reference_sha256': REFERENCE, 'candidate_sha256': candidate_sha, 'policy': POLICY,
              'trace_count': len(rows), 'all_passed': not failures, 'failures': failures, 'traces': rows}
    json_file(args.output, result)
    print(f'{len(rows)} tensors; {len(failures)} fail the unchanged policy')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
