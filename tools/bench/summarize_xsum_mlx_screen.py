"""Summarize an XSum Metal/Python five-trial screen; this is not a gate."""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def flatten(rows):
    return [item for row in rows for item in row] if rows and isinstance(rows[0], list) else rows


def maximum_rss(text):
    matches = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', text, re.MULTILINE)
    if len(matches) != 1 or int(matches[0]) <= 0:
        raise ValueError('missing or ambiguous process RSS')
    return int(matches[0])


def summarize(directory):
    before = directory / 'perf-inputs.sha256'
    if before.read_bytes() != (directory / 'perf-postflight.sha256').read_bytes():
        raise ValueError('screen source hashes changed')
    summary = {'scope': 'one process per runtime and batch; two warmups and five trials; not a gate',
               'source_receipt_sha256': digest(before), 'batches': {}}
    for batch in (1, 4, 32):
        data = {runtime: json.loads((directory / f'{runtime}-{batch}.json').read_bytes())
                for runtime in ('rust', 'python')}
        if any(value.get('batch') != batch or value.get('warmups') != 2
               or len(value.get('trials_seconds', [])) != 5
               or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in value['trials_seconds'])
               for value in data.values()):
            raise ValueError('incomplete timing data')
        for field, width in (('codes', 200), ('ids', 48)):
            a, b = (flatten(data[runtime][field]) for runtime in ('rust', 'python'))
            if len(a) != batch * width or a != b:
                raise ValueError(f'{field} mismatch at batch {batch}')
        row = {}
        for runtime, value in data.items():
            row[runtime] = {'median_ms': statistics.median(value['trials_seconds']) * 1000,
                            'peak_rss_bytes': maximum_rss((directory / f'{runtime}-{batch}.time').read_text()),
                            'mlx_memory': value['mlx_memory'],
                            'output_sha256': digest(directory / f'{runtime}-{batch}.json')}
        summary['batches'][str(batch)] = row
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = summarize(args.directory)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
