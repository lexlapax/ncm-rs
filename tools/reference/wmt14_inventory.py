"""Inspect pinned WMT14 checkpoint metadata on CPU; this does not export a model."""
import argparse
import collections
import json
from pathlib import Path

import torch

from tinystories_artifact import TRAINING, verify_input

INPUTS = {
    'generator.pt': (480582539, 'ae580cbbb400a27b30c616634745d4c9cfa8b66806f936c6c26041e5b9640d4f'),
    'tokenizer.pt': (645790351, '74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702'),
    'config.yaml': (2177, '6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72'),
}


def inventory(source):
    for name, (size, digest) in INPUTS.items():
        verify_input(source / name, size, digest)
    result = {}
    for name in ('generator.pt', 'tokenizer.pt'):
        obj = torch.load(source / name, map_location='cpu', weights_only=True, mmap=True)
        if not isinstance(obj, dict) or set(obj) != {'model'} or not isinstance(obj['model'], dict):
            raise ValueError(f'unexpected checkpoint structure: {name}')
        tensors = obj['model']
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in tensors.items()):
            raise ValueError(f'non-tensor checkpoint entry: {name}')
        result[name] = {
            'count': len(tensors),
            'bytes': sum(t.numel() * t.element_size() for t in tensors.values()),
            'dtypes': dict(sorted(collections.Counter(str(t.dtype) for t in tensors.values()).items())),
            'training_name_pattern_matches': sorted(k for k in tensors if TRAINING.fullmatch(k)),
            'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                        for k, v in sorted(tensors.items())},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.source), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
