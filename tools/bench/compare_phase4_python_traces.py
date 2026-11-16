"""Compare retained Python generator tensors without loading GPU libraries."""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def header(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('truncated safetensors header')
        length = struct.unpack('<Q', length_bytes)[0]
        if not 2 <= length <= min(size - 8, 1024 * 1024):
            raise ValueError('invalid safetensors header length')
        metadata = json.loads(stream.read(length))
    return metadata, 8 + length, size


def tensor(path, meta, base, size, key):
    item = meta[key]
    shape = item['shape']
    offsets = item['data_offsets']
    if (item['dtype'] != 'F32' or not shape or
            any(type(x) is not int or x <= 0 for x in shape) or
            len(offsets) != 2 or any(type(x) is not int for x in offsets)):
        raise ValueError(f'invalid tensor metadata: {key}')
    start, end = offsets
    if start < 0 or end < start or base + end > size or end - start != math.prod(shape) * 4:
        raise ValueError(f'invalid tensor extent: {key}')
    return np.memmap(path, dtype='<f4', mode='r', offset=base + start, shape=tuple(shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--adapted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    watched = [Path(__file__).resolve(), args.upstream.resolve(), args.adapted.resolve()]
    before = {str(p): digest(p) for p in watched}
    a, abase, asize = header(args.upstream)
    b, bbase, bsize = header(args.adapted)
    def is_logit(key):
        return key == 'generator.final_logits' or (key.startswith('generator.level_') and key.endswith('.logits'))

    keys = sorted(k for k in a if is_logit(k))
    if len(keys) != 12 or set(keys) != {k for k in b if is_logit(k)}:
        raise ValueError('unexpected generator logits inventory')
    result = {}
    for key in keys:
        x = tensor(args.upstream, a, abase, asize, key)
        y = tensor(args.adapted, b, bbase, bsize, key)
        if x.shape != y.shape or len(x.shape) != 3:
            raise ValueError(f'tensor shape mismatch: {key}')
        rows_x = x.reshape(-1, x.shape[-1])
        rows_y = y.reshape(-1, y.shape[-1])
        outside = decisions = 0
        max_normalized = max_abs = 0.0
        for start in range(0, len(rows_x), 128):
            ref = np.asarray(rows_x[start:start + 128], dtype=np.float64)
            other = np.asarray(rows_y[start:start + 128], dtype=np.float64)
            if not np.isfinite(ref).all() or not np.isfinite(other).all():
                raise ValueError(f'nonfinite tensor: {key}')
            diff = np.abs(other - ref)
            normalized = diff / (5e-4 + 2e-4 * np.abs(ref))
            outside += int(np.count_nonzero(normalized > 1))
            max_normalized = max(max_normalized, float(normalized.max()))
            max_abs = max(max_abs, float(diff.max()))
            decisions += int(np.count_nonzero(np.argmax(ref, axis=1) != np.argmax(other, axis=1)))
        result[key] = {'elements': int(x.size), 'outside_band': outside,
                       'max_normalized': max_normalized, 'max_abs': max_abs,
                       'argmax_differences': decisions}
    after = {str(p): digest(p) for p in watched}
    if before != after:
        raise ValueError('input or source changed during comparison')
    payload = {'upstream': str(args.upstream), 'adapted': str(args.adapted),
               'input_sha256': before, 'atol': 5e-4, 'rtol': 2e-4,
               'scope': 'retained Python generator logits; adapted trace is not independent for other tensors',
               'results': result}
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
