"""Inspect the pinned LM1B checkpoint family on CPU; no model export or inference."""
import argparse
import collections
import json
from pathlib import Path

import torch

from tinystories_artifact import digest_file, verify_input

CHECKPOINTS = {
    'small/generator.pt': (390138443, 'b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe'),
    'medium/generator.pt': (1268014283, '5aa4621472c2e658ab343e50d98abc808e4c2c6dd26eaf8535897b6a540c1604'),
    'large/generator.pt': (2901184267, 'ec34d6c56a81f428303b2fa3a8e1615ceee81b7dec7f59380427f46808ec224c'),
    'tokenizer.pt': (597840331, 'd88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c'),
}


def describe(value):
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError('checkpoint metadata must be a string-keyed dictionary')
    if all(isinstance(v, torch.Tensor) for v in value.values()):
        storage = collections.defaultdict(list)
        for name, tensor in value.items():
            storage[tensor.untyped_storage().data_ptr()].append(name)
        return {'count': len(value), 'bytes': sum(v.numel() * v.element_size() for v in value.values()),
                'dtypes': dict(sorted(collections.Counter(str(v.dtype) for v in value.values()).items())),
                'shared_storage': sorted(sorted(names) for names in storage.values() if len(names) > 1),
                'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                            for k, v in sorted(value.items())}}
    return {key: describe(child) for key, child in sorted(value.items())}


def inventory(source):
    results = {}
    for name, (size, sha) in CHECKPOINTS.items():
        path = source / name
        verify_input(path, size, sha)
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        results[name] = describe(value)
        verify_input(path, size, sha)
        del value
    results['config_observed'] = {name: {'bytes': (source / name / 'config.yaml').stat().st_size,
                                        'sha256': digest_file(source / name / 'config.yaml')}
                                  for name in ('small', 'medium', 'large')}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.source), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
