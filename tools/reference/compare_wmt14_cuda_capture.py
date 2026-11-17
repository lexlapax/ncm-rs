"""Compare a bounded contiguous WMT14 CUDA output capture."""
import argparse
import json
from pathlib import Path
from tinystories_artifact import digest_file, json_file


def compare(reference, actual, start, count):
    if (type(start) is not int or type(count) is not int or start < 0 or count < 1 or
            start + count > 3003 or len(reference) != count or len(actual) != count):
        raise ValueError('invalid range or row count')
    changes = {'code_differences': 0, 'id_differences': 0, 'code_rows': [], 'id_rows': []}
    for index, (left, right) in enumerate(zip(reference, actual, strict=True), start):
        if left.keys() != {'row', 'codes', 'raw_ids'} or right.keys() != left.keys() or left['row'] != index or right['row'] != index:
            raise ValueError('row schema or order differs')
        for key, width, limit, counter, rows in [('codes', 313, 4096, 'code_differences', 'code_rows'),
                                                 ('raw_ids', 64, 151680, 'id_differences', 'id_rows')]:
            if any(type(values) is not list or len(values) != width or
                   any(type(x) is not int or not 0 <= x < limit for x in values) for values in (left[key], right[key])):
                raise ValueError('invalid output values')
            delta = sum(a != b for a, b in zip(left[key], right[key], strict=True))
            changes[counter] += delta
            if delta:
                changes[rows].append(index)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--candidate-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists() or digest_file(args.reference) != args.reference_sha256 or digest_file(args.candidate) != args.candidate_sha256:
        raise ValueError('output exists or capture hash differs')
    left = json.loads(args.reference.read_bytes())
    right = json.loads(args.candidate.read_bytes())
    if left['batch_size'] != 4 or right['batch_size'] != 4 or (left['start'], left['count']) != (right['start'], right['count']):
        raise ValueError('batching or range differs')
    result = {'scope': 'bounded CUDA output comparison; scoring and performance are separate',
              'start': left['start'], 'count': left['count'], 'reference_sha256': args.reference_sha256,
              'candidate_sha256': args.candidate_sha256,
              **compare(left['outputs'], right['outputs'], left['start'], left['count'])}
    json_file(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
