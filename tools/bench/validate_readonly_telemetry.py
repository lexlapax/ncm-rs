"""Offline CPU/GPU content and receipt-window checks for the read-only protocol.

No collection, acceptance decision, live provenance or per-call clock inference.
Caller must independently bind source/UUID/timezone and raw bytes to launch receipts.
One-second samples bracket windows; they do not continuously observe those windows.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from readonly_trace_preflight import require
from readonly_trace_sequence import ORDER
from readonly_trace_telemetry import FILE_LIMIT, GPU_HEADER, validate_cpu_prefix
from sample_linux_cpu import TICK_NAMES, activity

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# datetime serializes microseconds; this is timestamp quantization, not a model
# numerical tolerance or an allowance for clock steps/sampling gaps.
TIME_QUANTIZATION_NS = 1000


def finite(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            'expected finite nonnegative number')
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'expected nonnegative integer')
    return value


def utc_ns(text):
    stamp = datetime.fromisoformat(text)
    require(stamp.utcoffset() is not None, 'timestamp must declare its UTC offset')
    delta = stamp.astimezone(timezone.utc) - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 1000


def reject_constant(_value):
    raise ValueError('nonfinite JSON constant')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def optional(row, parser):
    require(isinstance(row, dict) and type(row.get('available')) is bool, 'invalid optional-data label')
    if row['available']:
        require(set(row) == {'available', 'value'} and isinstance(row['value'], str),
                'invalid optional-data value')
        parser(row['value'])
        return True
    require(set(row) == {'available', 'error'} and isinstance(row['error'], str) and row['error'],
            'unavailable optional data needs an error label')
    return False


def frequency(text):
    require(text.isdecimal() and int(text) > 0, 'invalid frequency')


def loadavg(text):
    parts = text.split()
    require(len(parts) == 5 and re.fullmatch(r'\d+/\d+', parts[3]) is not None
            and parts[4].isdecimal(), 'invalid loadavg')
    for value in parts[:3]:
        finite(float(value))
    running, total = map(int, parts[3].split('/'))
    require(0 <= running <= total and total > 0, 'invalid loadavg task counts')


def pressure(text):
    seen = set()
    for line in text.splitlines():
        parts = line.split()
        require(len(parts) == 5 and parts[0] in ('some', 'full') and parts[0] not in seen,
                'invalid pressure row')
        seen.add(parts[0])
        values = dict(part.split('=', 1) for part in parts[1:])
        require(set(values) == {'avg10', 'avg60', 'avg300', 'total'}, 'invalid pressure fields')
        for key in ('avg10', 'avg60', 'avg300'):
            require(finite(float(values[key])) <= 100, 'invalid pressure average')
        require(values['total'].isdecimal(), 'invalid pressure total')
    require('some' in seen, 'missing pressure some row')


def cpu_records(raw, expected_source):
    require(isinstance(expected_source, str) and re.fullmatch(r'[0-9a-f]{64}', expected_source) is not None,
            'expected reviewed CPU source SHA-256')
    require(isinstance(raw, bytes) and 0 < len(raw) <= FILE_LIMIT and raw.endswith(b'\n'),
            'CPU telemetry is oversized, empty or truncated')
    rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
            for line in raw.splitlines()]
    require(len(rows) == 903, 'expected metadata, 901 samples and completion')
    metadata, samples, completion = rows[0], rows[1:-1], rows[-1]
    validate_cpu_prefix([json.dumps(metadata), json.dumps(samples[0])], expected_source)
    require(metadata['diagnostic_only'] is True and metadata['smt_siblings'] == [2, 3]
            and integer(metadata['tick_rate_hz'], 1), 'CPU metadata/topology mismatch')
    optional(metadata['governor'], lambda text: require(bool(text), 'empty governor'))
    optional(metadata['scaling_driver'], lambda text: require(bool(text), 'empty scaling driver'))
    require(completion['kind'] == 'completion' and completion['status'] == 'complete'
            and integer(completion['samples']) == 901
            and finite(completion['actual_elapsed_seconds']) >= 900, 'incomplete CPU telemetry')
    previous = None
    stamps = []
    missing = {'frequency_khz': [], 'loadavg': [], 'pressure': []}
    unavailable_activity = []
    clock_low, clock_high = [], []
    origin_low, origin_high = [], []
    for index, row in enumerate(samples):
        require(row['kind'] == 'sample' and integer(row['index']) == index
                and finite(row['scheduled_elapsed_seconds']) == index, 'CPU sample index/schedule mismatch')
        start, end = integer(row['monotonic_start_ns']), integer(row['monotonic_end_ns'])
        require(start <= end, 'reversed snapshot read interval')
        stamp = utc_ns(row['utc'])
        stamps.append(stamp)
        clock_low.append(stamp - end - TIME_QUANTIZATION_NS)
        clock_high.append(stamp - start + TIME_QUANTIZATION_NS)
        lateness_ns = finite(row['lateness_seconds']) * 1e9
        # Lateness is read after snapshot end and before the next snapshot. The
        # sampler's unrecorded schedule origin must fit all these bounds.
        origin_low.append(end - index * 10**9 - lateness_ns - TIME_QUANTIZATION_NS)
        if index < 900:
            origin_high.append(integer(samples[index + 1]['monotonic_start_ns'])
                               - index * 10**9 - lateness_ns + TIME_QUANTIZATION_NS)
        ticks = row['cpu_ticks']
        require(isinstance(ticks, dict) and {'cpu', 'cpu0', 'cpu2', 'cpu3'} <= ticks.keys()
                and all(re.fullmatch(r'cpu(?:\d+)?', key) for key in ticks), 'CPU inventory mismatch')
        for values in ticks.values():
            require(isinstance(values, dict) and set(values) == set(TICK_NAMES), 'CPU tick fields changed')
            for value in values.values():
                integer(value)
        counters = row['system_counters']
        require(set(counters) == {'ctxt', 'procs_running', 'procs_blocked'}, 'system counters changed')
        for value in counters.values():
            integer(value)
        expected_activity = None if previous is None else activity(previous['cpu_ticks'], ticks)
        # Canonical JSON comparison rejects bool-for-integer/float-for-integer substitutions too.
        require(json.dumps(row['activity_since_previous'], sort_keys=True, allow_nan=False)
                == json.dumps(expected_activity, sort_keys=True, allow_nan=False), 'CPU activity arithmetic mismatch')
        if previous is not None:
            require(previous['monotonic_end_ns'] <= start and stamps[-2] < stamp,
                    'CPU samples are not time-ordered')
            require(ticks.keys() == previous['cpu_ticks'].keys(), 'CPU inventory changed during collection')
            require(counters['ctxt'] >= previous['system_counters']['ctxt'], 'context-switch counter decreased')
            unavailable_activity.extend({'index': index, 'cpu': cpu, 'details': value}
                                        for cpu, value in expected_activity.items() if not value['available'])
        require(set(row['frequency_khz']) == {'0', '2', '3'} and set(row['pressure']) == {'cpu', 'memory', 'io'},
                'optional CPU inventory changed')
        for cpu, value in row['frequency_khz'].items():
            if not optional(value, frequency):
                missing['frequency_khz'].append({'index': index, 'cpu': cpu, 'error': value['error']})
        if not optional(row['loadavg'], loadavg):
            missing['loadavg'].append({'index': index, 'error': row['loadavg']['error']})
        for kind, value in row['pressure'].items():
            if not optional(value, pressure):
                missing['pressure'].append({'index': index, 'kind': kind, 'error': value['error']})
        previous = row
    require(max(clock_low) <= min(clock_high), 'UTC/monotonic mapping inconsistent across read intervals')
    origin_high.append(samples[0]['monotonic_start_ns'])
    origin_low.append(samples[-1]['monotonic_end_ns'] - completion['actual_elapsed_seconds'] * 1e9)
    require(max(origin_low) <= min(origin_high), 'lateness/completion incompatible with sampler schedule')
    return {'metadata': metadata, 'completion': completion, 'samples': samples, 'utc_ns': stamps,
            'missing_optional': missing, 'unavailable_activity': unavailable_activity}


def gpu_records(raw, expected_uuid, timezone_name):
    require(isinstance(expected_uuid, str)
            and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', expected_uuid) is not None,
            'expected reviewed GPU UUID')
    require(isinstance(raw, bytes) and 0 < len(raw) <= FILE_LIMIT and raw.endswith(b'\n'),
            'GPU telemetry is oversized, empty or truncated')
    zone = ZoneInfo(timezone_name)  # caller supplies launch-bound timezone, never infer from CSV
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8')), skipinitialspace=True, strict=True))
    require(len(rows) >= 3 and rows[0] == GPU_HEADER, 'GPU header/inventory mismatch')
    specs = [('utilization_percent', ' %', True, 100), ('memory_mib', ' MiB', True, None),
             ('temperature_c', '', True, None), ('power_w', ' W', False, None),
             ('sm_mhz', ' MHz', True, None), ('memory_mhz', ' MHz', True, None)]
    result = []
    for index, row in enumerate(rows[1:]):
        require(len(row) == len(GPU_HEADER) and row[1] == expected_uuid, 'GPU UUID/column mismatch')
        local = datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f')
        choices = set()
        for fold in (0, 1):
            stamp = local.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
            if stamp.astimezone(zone).replace(tzinfo=None) == local:
                choices.add(stamp)
        require(len(choices) == 1, 'ambiguous/nonexistent local GPU timestamp')
        stamp = choices.pop()
        normalized = {'index': index, 'utc': stamp.isoformat(), 'utc_ns': utc_ns(stamp.isoformat())}
        for text, (key, unit, integral, maximum) in zip(row[2:], specs):
            if text in ('N/A', '[N/A]', '[Not Supported]'):
                normalized[key] = {'available': False, 'reason': text}
                continue
            require((not unit or text.endswith(unit)), 'GPU units mismatch')
            number = text[:-len(unit)] if unit else text
            require(re.fullmatch(r'\d+' if integral else r'\d+(?:\.\d+)?', number) is not None,
                    'invalid GPU numeric field')
            value = int(number) if integral else float(number)
            finite(value)
            require(maximum is None or value <= maximum, 'GPU value outside supported range')
            normalized[key] = {'available': True, 'value': value}
        require(not result or result[-1]['utc_ns'] < normalized['utc_ns'], 'GPU timestamps not increasing')
        result.append(normalized)
    return result


def validate(cpu_raw, gpu_raw, jobs, *, expected_cpu_source, gpu_uuid, timezone_name):
    """Validate supplied telemetry and all twelve receipt windows, not live producers."""
    cpu = cpu_records(cpu_raw, expected_cpu_source)
    gpu = gpu_records(gpu_raw, gpu_uuid, timezone_name)
    require(isinstance(jobs, list) and len(jobs) == len(ORDER), 'twelve receipt windows required')
    windows, pids = [], set()
    last_end = None
    for ordinal, (condition, job) in enumerate(zip(ORDER, jobs), 1):
        require(integer(job['ordinal'], 1) == ordinal and job['condition'] == condition, 'job order mismatch')
        receipt = job['receipt']
        require(receipt['status'] == 'complete' and receipt['issues'] == []
                and receipt['direct_child_reaped'] is True and receipt['capture_eof'] is True
                and type(receipt['returncode']) is int and receipt['returncode'] == 0
                and receipt['interrupt_signal'] is None, 'incomplete receipt window')
        pid = integer(receipt['pid'], 1)
        require(pid not in pids, 'duplicate process window')
        pids.add(pid)
        start, end = utc_ns(receipt['started_utc']), utc_ns(receipt['finished_utc'])
        require(start < end and (last_end is None or last_end <= start), 'unordered/overlapping job windows')
        require(cpu['utc_ns'][0] <= start < end <= cpu['utc_ns'][-1]
                and gpu[0]['utc_ns'] <= start < end <= gpu[-1]['utc_ns'], 'telemetry does not bracket job')
        cpu_intervals = [i for i in range(1, len(cpu['utc_ns']))
                         if cpu['utc_ns'][i-1] < end and cpu['utc_ns'][i] > start]
        gpu_intervals = [i for i in range(1, len(gpu))
                         if gpu[i-1]['utc_ns'] < end and gpu[i]['utc_ns'] > start]
        windows.append({'ordinal': ordinal, 'condition': condition, 'pid': pid,
                        'started_utc': receipt['started_utc'], 'finished_utc': receipt['finished_utc'],
                        'cpu_interval_end_indices': cpu_intervals, 'gpu_interval_end_indices': gpu_intervals,
                        'gpu_sample_indices_inside': [r['index'] for r in gpu if start <= r['utc_ns'] <= end]})
        last_end = end
    return {'schema_version': 1, 'diagnostic_only': True, 'telemetry_contents_validated': True,
            'receipt_windows_bracketed': True, 'launch_provenance_verified': False,
            'continuous_gpu_observation_verified': False, 'performance_acceptance_evaluated': False,
            'cpu_sha256': hashlib.sha256(cpu_raw).hexdigest(), 'gpu_sha256': hashlib.sha256(gpu_raw).hexdigest(),
            'declared_cpu_source_sha256': expected_cpu_source, 'declared_gpu_uuid': gpu_uuid,
            'jobs_canonical_sha256': hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(',', ':'),
                                                              allow_nan=False).encode()).hexdigest(),
            'cpu_metadata': cpu['metadata'], 'cpu_completion': cpu['completion'],
            'missing_cpu_optional': cpu['missing_optional'], 'unavailable_cpu_activity': cpu['unavailable_activity'],
            'cpu_lateness_seconds': [r['lateness_seconds'] for r in cpu['samples']],
            'cpu_read_seconds': [(r['monotonic_end_ns'] - r['monotonic_start_ns']) / 1e9 for r in cpu['samples']],
            'cpu_sample_gaps_seconds': [(b-a) / 1e9 for a, b in zip(cpu['utc_ns'], cpu['utc_ns'][1:])],
            'gpu_timezone': timezone_name, 'gpu_samples': gpu,
            'gpu_sample_gaps_seconds': [(b['utc_ns'] - a['utc_ns']) / 1e9 for a, b in zip(gpu, gpu[1:])],
            'gpu_first_relative_to_cpu_first_seconds': (gpu[0]['utc_ns'] - cpu['utc_ns'][0]) / 1e9,
            'gpu_last_relative_to_cpu_last_seconds': (gpu[-1]['utc_ns'] - cpu['utc_ns'][-1]) / 1e9,
            'jobs': windows,
            'scope': 'Serialized contents, CPU counter arithmetic/schedule and coarse receipt bracketing only. '
                     'All gaps/unavailable values retained, not evidence of quietness or per-trial clocks. '
                     'No inferred samples, gap exclusions, new acceptance limits, lifetime/producer proof '
                     'or complete supervisor/collector/report/manifest binding.'}


def main():
    # Reuse the existing bounded, no-follow input reader; import here so future
    # analyzer composition can import validate() without a module-import cycle.
    from analyze_readonly_trace import read_bounded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cpu', type=Path)
    parser.add_argument('gpu', type=Path)
    parser.add_argument('sequence_result', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cpu-source-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--timezone', required=True)
    args = parser.parse_args()
    raw = read_bounded(args.sequence_result, FILE_LIMIT)
    sequence = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    require(type(sequence['schema_version']) is int and sequence['schema_version'] == 1
            and sequence['status'] == 'complete' and sequence['stop_reason'] is None,
            'incomplete sequence result')
    result = validate(read_bounded(args.cpu, FILE_LIMIT), read_bounded(args.gpu, FILE_LIMIT), sequence['jobs'],
                      expected_cpu_source=args.cpu_source_sha256, gpu_uuid=args.gpu_uuid,
                      timezone_name=args.timezone)
    result['sequence_result_sha256'] = hashlib.sha256(raw).hexdigest()
    result['tool_source_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
            'validate_readonly_telemetry.py', 'sample_linux_cpu.py', 'readonly_trace_telemetry.py',
            'readonly_trace_sequence.py', 'analyze_readonly_trace.py')}
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
