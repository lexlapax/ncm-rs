"""Validate sparse Rust-only outlier details against retained overlap summaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


FIELDS = ('both_outside', 'rust_only_outside', 'python_only_outside',
          'neither_outside', 'rust_python_different_bits')


def f32(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('nonfinite or invalid recorded float')
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('invalid recorded FP32 value') from error


def validate_pair(result, previous):
    keys = {key for key, value in previous.items()
            if isinstance(value, dict) and 'cross_runtime_overlap' in value}
    found = {key for key, value in result.items()
             if isinstance(value, dict) and 'cross_runtime_overlap' in value}
    if len(keys) != 12 or keys != found:
        raise ValueError('unexpected generator tensor inventory')
    details = []
    for key in sorted(keys):
        current, old = result[key], previous[key]
        overlap, expected = current['cross_runtime_overlap'], old['cross_runtime_overlap']
        if any(type(overlap.get(field)) is not int or overlap[field] < 0
               or overlap[field] != expected.get(field) for field in FIELDS):
            raise ValueError(f'overlap counts differ: {key}')
        count = current['elements']
        if type(count) is not int or count <= 0 or count != old['elements']:
            raise ValueError(f'invalid element count: {key}')
        if overlap['rust_python_different_bits'] > count:
            raise ValueError(f'invalid bit-difference count: {key}')
        if sum(overlap[field] for field in FIELDS[:4]) != count:
            raise ValueError(f'overlap count does not cover tensor: {key}')
        rows = overlap['rust_only_values']
        if not isinstance(rows, list) or len(rows) != overlap['rust_only_outside'] or len(rows) > 128:
            raise ValueError(f'invalid sparse outlier count: {key}')
        seen = set()
        for row in rows:
            offset = row['flat_offset']
            if type(offset) is not int or not 0 <= offset < count or offset in seen:
                raise ValueError(f'invalid sparse outlier offset: {key}')
            seen.add(offset)
            cuda, rust, python = (f32(row[field]) for field in ('cuda', 'rust_mlx', 'python_mlx'))
            limit = 5e-4 + 2e-4 * abs(cuda)
            rust_error, python_error = abs(rust - cuda), abs(python - cuda)
            for field, calculated in (('band_limit', limit), ('rust_abs_error', rust_error),
                                      ('python_abs_error', python_error)):
                value = row[field]
                if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(
                        value, calculated, rel_tol=1e-14, abs_tol=1e-18):
                    raise ValueError(f'inconsistent {field}: {key}')
            if not rust_error > limit >= python_error:
                raise ValueError(f'not a Rust-only crossing: {key}')
            details.append({'tensor': key, 'flat_offset': offset, 'rust_ratio': rust_error / limit,
                            'rust_overage': rust_error - limit, 'rust_python_abs': abs(rust - python)})
    return details


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--prior', type=Path, action='append', required=True)
    args = parser.parse_args()
    if len(args.result) != 5 or len(args.prior) != 5:
        raise ValueError('exactly five ordered result/prior pairs are required')
    watched = [Path(__file__).resolve(), *(path.resolve() for path in args.result + args.prior)]
    if len(set(watched)) != len(watched):
        raise ValueError('result, prior and source paths must be distinct')
    before = {str(path): digest(path) for path in watched}
    records = []
    for seed, (result, prior) in enumerate(zip(args.result, args.prior), start=100):
        for item in validate_pair(json.loads(result.read_bytes()), json.loads(prior.read_bytes())):
            records.append({'seed': seed, **item})
    if before != {str(path): digest(path) for path in watched}:
        raise ValueError('source or result changed during validation')
    print(json.dumps({'records': records, 'count': len(records), 'input_sha256': before,
                      'scope': 'validated sparse values and retained overlap counts; not numerical acceptance'},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
