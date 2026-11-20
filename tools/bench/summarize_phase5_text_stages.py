"""Summarize host-only CLI tokenizer-stage diagnostic logs without text or paths."""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

STAGES = ('construct_seconds', 'encode_seconds', 'decode_seconds')


def parse(profile, log, time_file):
    lines = log.decode().splitlines()
    payloads = [line.split('TEXT_STAGE_TIMINGS ', 1)[1] for line in lines if 'TEXT_STAGE_TIMINGS ' in line]
    if len(payloads) != 1 or not any('test result: ok. 1 passed;' in line for line in lines):
        raise ValueError('missing successful diagnostic')
    result = json.loads(payloads[0])
    if (set(result) != {'profile', 'warmups', 'trials', *STAGES} or result['profile'] != profile
            or result['warmups'] != 5 or result['trials'] != 30):
        raise ValueError('diagnostic identity differs')
    for stage in STAGES:
        values = result[stage]
        if (not isinstance(values, list) or len(values) != 30
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)):
            raise ValueError('invalid timing samples')
    matches = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', time_file.decode())
    if len(matches) != 1 or int(matches[0]) <= 0:
        raise ValueError('invalid peak RSS')
    return result, int(matches[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    export = {'scope': 'host tokenizer stages on already loaded artifacts; not cold CLI or release qualification',
              'warmups_per_process': 5, 'trials_per_process': 30, 'processes_per_profile': 3, 'profiles': {}}
    for profile in ('xsum', 'wmt14'):
        records, hashes, rss = [], {}, []
        for rep in range(3):
            log_name, time_name = f'{profile}-r{rep}.log', f'{profile}-r{rep}.time'
            log, time_file = (args.directory / log_name).read_bytes(), (args.directory / time_name).read_bytes()
            record, peak = parse(profile, log, time_file)
            records.append(record); rss.append(peak)
            hashes[log_name] = hashlib.sha256(log).hexdigest()
            hashes[time_name] = hashlib.sha256(time_file).hexdigest()
        stages = {}
        for stage in STAGES:
            values = [value for record in records for value in record[stage]]
            stages[stage] = {'per_process': [record[stage] for record in records],
                             'pooled_median_seconds': statistics.median(values),
                             'pooled_p95_seconds': sorted(values)[math.ceil(.95 * len(values)) - 1],
                             'process_medians_seconds': [statistics.median(record[stage]) for record in records]}
        export['profiles'][profile] = {'stages': stages, 'peak_rss_kib': rss, 'input_sha256': hashes}
    print(json.dumps(export, sort_keys=True, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
