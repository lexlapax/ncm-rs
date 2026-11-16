"""Inspect pinned LM1B inference unigram metadata on CPU; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from lm1b_inventory import CHECKPOINTS
from tinystories_artifact import tensor_bytes, verify_input


def inspect(source):
    result = {}
    for size in ('small', 'medium', 'large'):
        name = f'{size}/generator.pt'
        path = source / name
        verify_input(path, *CHECKPOINTS[name])
        obj = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if set(obj) != {'model', 'inference'} or set(obj['inference']) != {'level0_unigram', 'token_unigram'}:
            raise ValueError('unexpected inference inventory')
        result[size] = {}
        for key, length in (('level0_unigram', 8192), ('token_unigram', 30528)):
            t = obj['inference'][key]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or tuple(t.shape) != (length,):
                raise ValueError('unexpected inference tensor')
            result[size][key] = {'sha256': hashlib.sha256(tensor_bytes(t)).hexdigest(),
                                 'finite': bool(torch.isfinite(t).all()),
                                 'negative_count': int((t < 0).sum()),
                                 'zero_count': int((t == 0).sum()),
                                 'sum_f64': float(t.double().sum()),
                                 'minimum': float(t.min()), 'maximum': float(t.max())}
        del obj
        verify_input(path, *CHECKPOINTS[name])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
