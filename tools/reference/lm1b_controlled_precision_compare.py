"""Compare controlled LM1B traces across strict and evaluator convolution settings."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from compare_lm1b_small_controlled_trace import REFERENCE, compare_tensor
from lm1b_small_controlled_trace import INPUTS, PREFIX_SHA
from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file

LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)


def compare(reference, candidate, diagnostic_level=7):
    with safe_open(reference, framework='pt', device='cpu') as left, safe_open(
            candidate, framework='pt', device='cpu') as right:
        names = set(left.keys())
        extra = set(right.keys()) - names
        if names - set(right.keys()) or any(not key.startswith(f'diagnostic.level_{diagnostic_level:02d}.') for key in extra):
            raise ValueError('unexpected controlled tensor inventory')
        comparisons = [compare_tensor(name, left.get_tensor(name), right.get_tensor(name))
                       for name in sorted(names)]
        codes_left = left.get_tensor('generator.generated_codes')
        codes_right = right.get_tensor('generator.generated_codes')
        if tuple(codes_left.shape) != (2, sum(LEVELS)):
            raise ValueError('unexpected code geometry')
        offsets = []
        offset = 0
        for level, size in enumerate(LEVELS):
            delta = codes_left[:, offset:offset + size] != codes_right[:, offset:offset + size]
            locations = torch.nonzero(delta)
            if locations.numel():
                row, position = locations[0].tolist()
                offsets.append({'level': level, 'mismatches': int(delta.sum()),
                                'first': {'row': row, 'position': position}})
            offset += size
    return {'compared_tensors': len(comparisons), 'extra_candidate_tensors': sorted(extra),
            'failures': [row['name'] for row in comparisons
                         if row['mismatches'] or row.get('argmax_mismatches', 0)],
            'code_differences': offsets, 'tensors': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('python_strict', type=Path)
    parser.add_argument('python_evaluator', type=Path)
    parser.add_argument('rust_strict', type=Path)
    parser.add_argument('rust_default', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('output must be new and under reports')
    torch.set_num_threads(1)
    reference = args.python_evaluator / 'trace.safetensors'
    report_path = args.python_evaluator / 'report.json'
    receipt = json.loads(report_path.read_bytes())
    paths = {'python_strict': args.python_strict, 'python_evaluator': reference,
             'python_evaluator_receipt': report_path, 'rust_strict': args.rust_strict,
             'rust_default': args.rust_default}
    before = {name: digest_file(path) for name, path in paths.items()}
    if (before['python_strict'] != REFERENCE or receipt['trace_sha256'] != before['python_evaluator']
            or receipt.get('controlled_seed', 11) != 11
            or receipt['upstream'] != UPSTREAM or receipt['code_reference_sha256'] != PREFIX_SHA
            or receipt['checkpoint_sha256'] != {name: sha for name, (_, sha) in INPUTS.items()}
            or receipt['tf32'] is not False or receipt['sdpa'] != 'math'
            or receipt.get('cudnn_convolution') != {'allow_tf32': True, 'benchmark': False, 'deterministic': False, 'legacy_api_requested': False}):
        raise ValueError('unexpected Python controlled reference')
    pairs = (('python_strict', 'rust_strict'), ('python_evaluator', 'rust_default'),
             ('python_evaluator', 'rust_strict'), ('python_strict', 'rust_default'),
             ('python_strict', 'python_evaluator'))
    results = {}
    for left, right in pairs:
        label = f'{left}__{right}'
        result = compare(paths[left], paths[right])
        results[label] = result
        print(label, result['compared_tensors'], len(result['failures']), result['code_differences'][:1])
    if before != {name: digest_file(path) for name, path in paths.items()}:
        raise ValueError('controlled input changed')
    json_file(out, {'scope': 'two development rows with common controlled draws', 'inputs_sha256': before,
                    'comparisons': results})


if __name__ == '__main__':
    main()
