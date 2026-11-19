"""Audit compact Phase 5 cold-screen aggregates against retained local captures.

This checks selected statistics and capture identity only; it does not establish
benchmark validity, model quality, GPU isolation, or release qualification.
Inputs and their parent directories must be trusted and stable during the audit.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

PROFILES = ('sudoku-easy', 'tinystories', 'xsum', 'wmt14', 'lm1b-small', 'lm1b-medium', 'lm1b-large')
FIELDS = ('median_seconds', 'p95_seconds_nearest_rank', 'median_absolute_deviation_seconds',
          'median_max_rss_kib', 'max_rss_kib')


def read_regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'not a regular file: {path}')
    return path.read_bytes()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def calculate(records, root):
    if not isinstance(records, list) or len(records) != 210:
        raise ValueError('expected 210 records')
    expected = {(g, o, c) for g in range(1, 4) for o in range(35) for c in ('baseline', 'candidate')}
    seen = set()
    output_hashes = set()
    measured = {'baseline': [], 'candidate': []}
    for row in records:
        g, o, c = row['group'], row['ordinal'], row['condition']
        if type(g) is not int or type(o) is not int or (g, o, c) not in expected or (g, o, c) in seen:
            raise ValueError('invalid or duplicate trial identity')
        seen.add((g, o, c))
        if type(row['warmup']) is not bool or row['warmup'] != (o < 5):
            raise ValueError('warmup identity differs')
        elapsed, rss = row['elapsed_seconds'], row['max_rss_kib']
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('invalid elapsed time')
        if type(rss) is not int or rss <= 0:
            raise ValueError('invalid RSS')
        output = root / f'{g}-{o:02d}-{c}.stdout'
        output_hash = sha(read_regular(output))
        if output_hash != row['output_sha256']:
            raise ValueError(f'stdout hash differs: {output}')
        output_hashes.add(output_hash)
        if not row['warmup']:
            measured[c].append((elapsed, rss))
    if seen != expected or len(output_hashes) != 1:
        raise ValueError('missing trials or nonidentical stdout')
    result = {}
    for condition, rows in measured.items():
        if len(rows) != 90:
            raise ValueError('expected 90 measured trials per condition')
        values = sorted(row[0] for row in rows)
        rss = [row[1] for row in rows]
        median = statistics.median(values)
        result[condition] = {'median_seconds': median,
                             'p95_seconds_nearest_rank': values[math.ceil(.95 * len(values)) - 1],
                             'median_absolute_deviation_seconds': statistics.median(abs(x - median) for x in values),
                             'median_max_rss_kib': statistics.median(rss), 'max_rss_kib': max(rss)}
    return result


def audit_one(name, entry, root, baseline_sha, candidate_sha):
    result_path, gpu_path = root / 'result.json', root / 'gpu.csv'
    raw, gpu = read_regular(result_path), read_regular(gpu_path)
    if sha(raw) != entry['raw_result_sha256'] or sha(gpu) != entry['gpu_csv_sha256']:
        raise ValueError(f'{name}: raw result or GPU CSV hash differs')
    data = json.loads(raw)
    if data['gpu_csv_sha256'] != entry['gpu_csv_sha256']:
        raise ValueError(f'{name}: recorded GPU CSV hash differs')
    watched = data['watched_sha256']
    for filename, expected in (('baseline-ncm', baseline_sha), ('ncm', candidate_sha)):
        if not any(Path(path).name == filename and value == expected for path, value in watched.items()):
            raise ValueError(f'{name}: recorded binary hash differs')
    values = calculate(data['records'], root)
    for condition in ('baseline', 'candidate'):
        if values[condition] != entry[condition]:
            raise ValueError(f'{name}: aggregate differs for {condition}')
        if any(data['summary'][condition][key] != values[condition][key] for key in FIELDS):
            raise ValueError(f'{name}: recorded summary differs for {condition}')
    if read_regular(result_path) != raw or read_regular(gpu_path) != gpu:
        raise ValueError(f'{name}: inputs changed during audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--aggregate-sha256', required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    raw = read_regular(args.aggregate)
    if sha(raw) != args.aggregate_sha256:
        raise ValueError('aggregate hash differs')
    aggregate = json.loads(raw)
    if aggregate['schema_version'] != 1 or set(aggregate['profiles']) != set(PROFILES):
        raise ValueError('unexpected aggregate profile inventory')
    if aggregate['samples_per_condition'] != 90:
        raise ValueError('unexpected sample count')
    for name in PROFILES:
        folder = 'phase5-cli-cold-sudoku-easy-v4' if name == 'sudoku-easy' else f'phase5-cli-cold-{name}-v1'
        audit_one(name, aggregate['profiles'][name], args.reports / folder,
                  aggregate['baseline_binary_sha256'], aggregate['candidate_binary_sha256'])
        print(f'{name}: 210 stdout hashes and 90 measured records per condition verified')
    if read_regular(args.aggregate) != raw:
        raise ValueError('aggregate changed during audit')
    print('Capture checks passed; benchmark, GPU and release qualification are not implied.')


if __name__ == '__main__':
    main()
