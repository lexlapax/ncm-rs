"""Export audited cold-request timing/RSS series, excluding model inputs and outputs.

This preserves measurements, not GPU telemetry or workload qualification. The raw
captures and aggregate must be trusted and stable; no generated data are exported.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import audit_phase5_cold_aggregates as audit

AGGREGATE_SHA = 'adff3aea3317dd3bd7af0886bf6d0d2d1dbf854f96286de7ec642e71248fd762'


def extract(records):
    if not isinstance(records, list) or len(records) != 210:
        raise ValueError('expected 210 trial records')
    index = {}
    for row in records:
        group, ordinal, condition = row['group'], row['ordinal'], row['condition']
        key = (group, ordinal, condition)
        if (type(group) is not int or group not in (1, 2, 3) or type(ordinal) is not int
                or not 0 <= ordinal < 35 or condition not in ('baseline', 'candidate') or key in index
                or type(row['warmup']) is not bool or row['warmup'] != (ordinal < 5)):
            raise ValueError('invalid trial identity')
        elapsed, rss = row['elapsed_seconds'], row['max_rss_kib']
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0 or type(rss) is not int or rss <= 0:
            raise ValueError('invalid timing or RSS')
        index[key] = (elapsed, rss)
    groups = []
    for group in range(1, 4):
        conditions = {}
        for condition in ('baseline', 'candidate'):
            rows = [index[(group, ordinal, condition)] for ordinal in range(5, 35)]
            conditions[condition] = {'elapsed_seconds': [row[0] for row in rows], 'max_rss_kib': [row[1] for row in rows]}
        groups.append(conditions)
    return groups


def summarize(groups):
    if not isinstance(groups, list) or len(groups) != 3:
        raise ValueError('expected three groups')
    result = {}
    for condition in ('baseline', 'candidate'):
        elapsed, rss = [], []
        for group in groups:
            if set(group) != {'baseline', 'candidate'} or set(group[condition]) != {'elapsed_seconds', 'max_rss_kib'}:
                raise ValueError('invalid group fields')
            times, memory = group[condition]['elapsed_seconds'], group[condition]['max_rss_kib']
            if len(times) != 30 or len(memory) != 30 or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in times
            ) or any(type(value) is not int or value <= 0 for value in memory):
                raise ValueError('invalid measured values')
            elapsed.extend(times); rss.extend(memory)
        ordered = sorted(elapsed)
        median = statistics.median(ordered)
        result[condition] = {'median_seconds': median,
                             'p95_seconds_nearest_rank': ordered[math.ceil(.95 * len(ordered)) - 1],
                             'median_absolute_deviation_seconds': statistics.median(abs(value - median) for value in ordered),
                             'median_max_rss_kib': statistics.median(rss), 'max_rss_kib': max(rss)}
    return result


def verify_export(export, aggregate):
    expected = {'schema_version', 'scope', 'aggregate_sha256', 'warmup_trials_omitted', 'pair_order', 'profiles'}
    if (not isinstance(export, dict) or set(export) != expected or export['schema_version'] != 1
            or export['aggregate_sha256'] != AGGREGATE_SHA or export['warmup_trials_omitted'] is not True
            or set(export['profiles']) != set(audit.PROFILES) or set(aggregate['profiles']) != set(audit.PROFILES)):
        raise ValueError('invalid export inventory')
    for name in audit.PROFILES:
        value, entry = export['profiles'][name], aggregate['profiles'][name]
        if (set(value) != {'raw_result_sha256', 'gpu_csv_sha256', 'groups'}
                or value['raw_result_sha256'] != entry['raw_result_sha256']
                or value['gpu_csv_sha256'] != entry['gpu_csv_sha256']
                or summarize(value['groups']) != {condition: entry[condition] for condition in ('baseline', 'candidate')}):
            raise ValueError(f'export differs: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', required=True, type=Path)
    parser.add_argument('--reports', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    raw = audit.read_regular(args.aggregate)
    if audit.sha(raw) != AGGREGATE_SHA:
        raise ValueError('aggregate identity differs')
    aggregate = json.loads(raw)
    if aggregate['schema_version'] != 1 or set(aggregate['profiles']) != set(audit.PROFILES):
        raise ValueError('aggregate inventory differs')
    result = {'schema_version': 1, 'scope': 'measured cold-request seconds and process RSS only; not release qualification',
              'aggregate_sha256': AGGREGATE_SHA, 'warmup_trials_omitted': True,
              'pair_order': 'three groups, ordinals 5..34; baseline and candidate arrays align by ordinal',
              'profiles': {}}
    for name in audit.PROFILES:
        folder = 'phase5-cli-cold-sudoku-easy-v4' if name == 'sudoku-easy' else f'phase5-cli-cold-{name}-v1'
        directory = args.reports / folder
        entry = aggregate['profiles'][name]
        audit.audit_one(name, entry, directory, aggregate['baseline_binary_sha256'], aggregate['candidate_binary_sha256'])
        raw_result = audit.read_regular(directory / 'result.json')
        if audit.sha(raw_result) != entry['raw_result_sha256']:
            raise ValueError('raw result changed')
        groups = extract(json.loads(raw_result)['records'])
        if summarize(groups) != {key: entry[key] for key in ('baseline', 'candidate')}:
            raise ValueError('sanitized summary differs')
        result['profiles'][name] = {'raw_result_sha256': entry['raw_result_sha256'],
                                    'gpu_csv_sha256': entry['gpu_csv_sha256'], 'groups': groups}
    verify_export(result, aggregate)
    if audit.read_regular(args.aggregate) != raw:
        raise ValueError('aggregate changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
