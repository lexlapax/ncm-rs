"""Compare a bounded WMT14 CUDA development screen; no quality claim."""
import argparse
import json
from pathlib import Path
from tinystories_artifact import digest_file, json_file

REFERENCE = 'f86979d47a76b789889c835fed5a7c3425768cf97ffc1d7ab19ab4daa998a852'


def compare(reference, actual):
    if len(reference) != 32 or len(actual) != 32:
        raise ValueError('expected 32 rows')
    changes = {'code_differences': 0, 'id_differences': 0, 'code_rows': [], 'id_rows': []}
    for index, (left, right) in enumerate(zip(reference, actual, strict=True)):
        if left.keys() != {'row', 'codes', 'raw_ids'} or right.keys() != left.keys() or left['row'] != index or right['row'] != index:
            raise ValueError('row schema or order differs')
        for key, count, limit, counter, rows in [('codes', 313, 4096, 'code_differences', 'code_rows'),
                                                 ('raw_ids', 64, 151680, 'id_differences', 'id_rows')]:
            if any(type(values) is not list or len(values) != count or
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
    args = parser.parse_args()
    if args.output.exists() or digest_file(args.reference) != REFERENCE:
        raise ValueError('output exists or upstream reference differs')
    left = json.loads(args.reference.read_bytes())
    right = json.loads(args.candidate.read_bytes())
    if left['batch_size'] != 4 or right['batch_size'] != 4:
        raise ValueError('batching differs')
    result = {'scope': '32-row CUDA output screen; not quality or performance', 'rows': 32,
              'reference_sha256': REFERENCE, 'candidate_sha256': digest_file(args.candidate),
              **compare(left['outputs'], right['outputs'])}
    json_file(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if result['code_differences'] or result['id_differences']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
