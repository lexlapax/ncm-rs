"""Audit and sanitize the retained CUDA text cold-screen measurements."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import phase5_text_cold_screen as screen

PINS = {
    'xsum': ('88c5e6e2af6eb69fc083f9c329b9ebe67a5379eb2122ff7e4768e4d5324893a7',
             '20247a8652f1259edb8b32a5f8ef19ad2b2f5c1d8cbb8e59829033e58653e53f'),
    'wmt14': ('eeed3c10a7f731453bdb12783bd7c5e2df3967dda254c37818b6a61b0d356efc',
              'e4c23d60d90073b5b8cd36f8dccfec76e38e12199b6366645963e9b4bb0431e0'),
}
COMPARISONS = (('old_id', 'new_id'), ('new_id', 'new_text'), ('new_text', 'new_text_decode'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def paired(records):
    by_key = {}
    for row in records:
        if not row['warmup']:
            by_key.setdefault((row['group'], row['ordinal']), {})[row['condition']] = row['elapsed_seconds']
    if len(by_key) != 90 or any(set(value) != set(screen.CONDITIONS) for value in by_key.values()):
        raise ValueError('incomplete paired measurements')
    result = {}
    for old, new in COMPARISONS:
        diffs = [value[new] - value[old] for _, value in sorted(by_key.items())]
        rng = random.Random(42)
        bootstrap = sorted(statistics.median(rng.choices(diffs, k=90)) for _ in range(10000))
        result[f'{new}_minus_{old}'] = {'median_seconds': statistics.median(diffs),
                                        'bootstrap_95_seconds': [bootstrap[249], bootstrap[9749]]}
    return result


def audit_run(name, directory):
    result_path, gpu_path = directory / 'result.json', directory / 'gpu.csv'
    raw, gpu = result_path.read_bytes(), gpu_path.read_bytes()
    if (digest(raw), digest(gpu)) != PINS[name]:
        raise ValueError('raw result or telemetry hash differs')
    data = json.loads(raw)
    if data['profile'] != name or data['gpu_csv_sha256'] != PINS[name][1]:
        raise ValueError('profile or telemetry differs')
    records = data['records']
    if screen.summarize(records) != data['summary']:
        raise ValueError('record summary differs')
    expected_order = [(group, ordinal, condition) for group in range(1, 4) for ordinal in range(35)
                      for condition in screen.order(group, ordinal)]
    if [(row['group'], row['ordinal'], row['condition']) for row in records] != expected_order:
        raise ValueError('execution order differs')
    for row in records:
        group, ordinal, condition = row['group'], row['ordinal'], row['condition']
        stem = f'{group}-{ordinal:02}-{condition}'
        stdout = (directory / f'{stem}.stdout').read_bytes()
        expected = screen.HASHES[name][condition == 'new_text_decode']
        if digest(stdout) != expected or row['output_sha256'] != expected:
            raise ValueError('stdout differs')
        if int((directory / f'{stem}.rss').read_text().strip()) != row['max_rss_kib']:
            raise ValueError('RSS differs')
    groups = []
    for group in range(1, 4):
        item = {}
        for condition in screen.CONDITIONS:
            rows = [row for row in records if row['group'] == group and row['condition'] == condition and not row['warmup']]
            item[condition] = {'elapsed_seconds': [row['elapsed_seconds'] for row in rows],
                               'max_rss_kib': [row['max_rss_kib'] for row in rows]}
        groups.append(item)
    if result_path.read_bytes() != raw or gpu_path.read_bytes() != gpu:
        raise ValueError('capture changed during audit')
    return {'raw_result_sha256': PINS[name][0], 'gpu_csv_sha256': PINS[name][1],
            'summary': data['summary'], 'paired': paired(records), 'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    result = {'scope': 'sanitized cold CLI timings and RSS only; not release qualification',
              'pairing': 'three groups, ordinal 5..34; arrays align by ordinal',
              'bootstrap': '10,000 paired median resamples using Python random.Random(42); descriptive only',
              'profiles': {name: audit_run(name, args.reports / f'phase5-{name}-text-cold-v1') for name in PINS}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
