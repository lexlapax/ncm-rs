"""Audit and sanitize retained LM1B empty-text cold-request measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import export_phase5_text_cold as common
import phase5_lm1b_text_cold as lm

PINS = {
    'small': ('d2198abd536cd3d5b810771349425c3075905e75efd1e0778b7f165883d4c766', '1a4d549f3d89681ac84ad190836ef5fe09fb34e5a023be2985302d3cbc97c52b'),
    'medium': ('0a105183f786def3dc6d5df5e88e010052b9055be102a7ba36747ed4e30a861a', '8f463d78e8814d7ddee5e617b6b8dfbdb33254f42536b014dc316259e7a6f6f4'),
    'large': ('fb4fb34cb3c58c7a4d2268777873aba529202fe757dbaae2a43f22f1a950a3d5', 'a67f8b317280be9b9f4479bd9124cc1e3d22fb56c40469a2e616cdd16e5bfcde'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_run(size, directory):
    result_path, gpu_path = directory / 'result.json', directory / 'gpu.csv'
    raw, gpu = result_path.read_bytes(), gpu_path.read_bytes()
    if (digest(raw), digest(gpu)) != PINS[size]:
        raise ValueError('raw result or telemetry differs')
    data = json.loads(raw)
    if data['size'] != size or data['gpu_csv_sha256'] != PINS[size][1]:
        raise ValueError('size or telemetry differs')
    records = data['records']
    if lm.cold.summarize(records) != data['summary']:
        raise ValueError('summary differs')
    expected_order = [(group, ordinal, condition) for group in range(1, 4) for ordinal in range(35)
                      for condition in lm.cold.order(group, ordinal)]
    if [(row['group'], row['ordinal'], row['condition']) for row in records] != expected_order:
        raise ValueError('trial order differs')
    for row in records:
        group, ordinal, condition = row['group'], row['ordinal'], row['condition']
        stem = f'{group}-{ordinal:02}-{condition}'
        expected = lm.HASHES[size][condition == 'new_text_decode']
        if digest((directory / f'{stem}.stdout').read_bytes()) != expected or row['output_sha256'] != expected:
            raise ValueError('stdout differs')
        if int((directory / f'{stem}.rss').read_text().strip()) != row['max_rss_kib']:
            raise ValueError('RSS differs')
    groups = []
    for group in range(1, 4):
        item = {}
        for condition in lm.cold.CONDITIONS:
            selected = [row for row in records if row['group'] == group and row['condition'] == condition and not row['warmup']]
            item[condition] = {'elapsed_seconds': [row['elapsed_seconds'] for row in selected],
                               'max_rss_kib': [row['max_rss_kib'] for row in selected]}
        groups.append(item)
    if result_path.read_bytes() != raw or gpu_path.read_bytes() != gpu:
        raise ValueError('capture changed during audit')
    return {'raw_result_sha256': PINS[size][0], 'gpu_csv_sha256': PINS[size][1],
            'summary': data['summary'], 'paired': common.paired(records), 'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    result = {'scope': 'sanitized LM1B cold CLI timings and RSS only; not release qualification',
              'pairing': 'three groups, ordinal 5..34; arrays align by ordinal',
              'bootstrap': '10,000 paired median resamples using Python random.Random(42); descriptive only',
              'sizes': {size: audit_run(size, args.reports / f'phase5-lm1b-{size}-text-cold-v1') for size in PINS}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
