"""Offline integration of fixed-sequence receipts, codec qualification and CUPTI joins.

Receipt consistency is not independent launch provenance. No GPU commands, timing
acceptance, telemetry validation or permission to collect are provided here.
"""
import argparse
import json
import os
from pathlib import Path
import stat

from bind_cupti_trials import bind_trials, digest, require
from phase2_summary import stats
from qualify_diagnostic_codec import qualify
from readonly_trace_sequence import (ORDER, CAPTURE_BYTES, TOTAL_BYTES, MIN_FREE_BYTES,
                                     CHILD_SECONDS, SEQUENCE_SECONDS)

REPORT_LIMIT = 16 * 1024 ** 2


def read_bounded(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        size = os.fstat(fd).st_size
        require(stat.S_ISREG(os.fstat(fd).st_mode) and 0 <= size <= limit,
                'invalid input file type or size')
        chunks, count = [], 0
        while chunk := os.read(fd, min(65536, limit + 1 - count)):
            chunks.append(chunk)
            count += len(chunk)
            require(count <= limit, 'input exceeds limit')
        require(count == size, 'input size changed while reading')
        return b''.join(chunks)
    finally:
        os.close(fd)


def analyze(root, sources):
    """Read only fixed filenames in a trusted directory; original paths are metadata.

    Archives may be relocated. Expected sources must be supplied from reviewed
    prelaunch evidence, not inferred from the reports being checked.
    """
    require(isinstance(sources, dict) and set(sources) == {'A', 'C'}, 'expected A/C source inventories')
    root = Path(root)
    plan_raw = read_bounded(root / 'sequence-plan.json', REPORT_LIMIT)
    result_raw = read_bounded(root / 'sequence-result.json', REPORT_LIMIT)
    plan, result = json.loads(plan_raw), json.loads(result_raw)
    require(type(plan['schema_version']) is int and plan['schema_version'] == 1
            and type(result['schema_version']) is int and result['schema_version'] == 1,
            'unsupported sequence schema')
    for key, value in {'sequence_seconds': SEQUENCE_SECONDS, 'child_seconds': CHILD_SECONDS,
                       'capture_limit_bytes': CAPTURE_BYTES, 'total_limit_bytes': TOTAL_BYTES,
                       'minimum_initial_free_bytes': MIN_FREE_BYTES}.items():
        require(type(plan[key]) is int and plan[key] == value, 'sequence limits changed')
    require(result['status'] == 'complete' and result['stop_reason'] is None,
            'incomplete sequence')
    require(len(plan['jobs']) == len(result['jobs']) == len(ORDER), 'expected all twelve jobs')
    groups = {c: [] for c in ('AP', 'CP', 'AT', 'CT')}
    joined, bindings, pids, canonical_reports = [], [], set(), set()
    total = 0
    for ordinal, (condition, job, recorded) in enumerate(zip(ORDER, plan['jobs'], result['jobs']), 1):
        require(job['ordinal'] == recorded['ordinal'] == ordinal
                and job['condition'] == recorded['condition'] == condition, 'job order mismatch')
        prefix = f'{ordinal:02d}-{condition}'
        receipt_raw = read_bounded(root / (prefix + '.receipt.json'), REPORT_LIMIT)
        receipt = json.loads(receipt_raw)
        require(receipt == recorded['receipt'], 'individual/sequence receipt mismatch')
        require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1,
                'unsupported child receipt')
        require(receipt['status'] == 'complete' and receipt['issues'] == []
                and type(receipt['returncode']) is int and receipt['returncode'] == 0
                and receipt['direct_child_reaped'] is True and receipt['capture_eof'] is True
                and receipt['interrupt_signal'] is None and receipt['group_kill'] in ('sent', 'no_group'),
                'incomplete child receipt')
        pid = receipt['pid']
        require(type(pid) is int and pid > 0 and pid not in pids, 'invalid or reused receipt PID')
        pids.add(pid)
        require(receipt['argv'] == job['argv'] and receipt['cwd'] == job['cwd'],
                'receipt command mismatch')
        report_binding = receipt['report_capture']
        require(isinstance(report_binding, dict) and report_binding['post_exit_hash_verified'] is True,
                'missing verified report capture')
        path = report_binding['path']
        require(isinstance(path, str) and Path(path).is_absolute()
                and Path(path).name == prefix + '.report.json'
                and path == job['report_path'] and job['argv'].count(path) == 1,
                'report path/command mismatch')
        require(receipt['timeout_seconds'] == CHILD_SECONDS
                and type(receipt['output_limit_bytes']) is int
                and receipt['output_limit_bytes'] == min(CAPTURE_BYTES, TOTAL_BYTES - total)
                and receipt['report_limit_bytes'] == REPORT_LIMIT, 'child limits changed')
        capture = read_bounded(root / (prefix + '.capture'), receipt['output_limit_bytes'])
        raw = read_bounded(root / (prefix + '.report.json'), REPORT_LIMIT)
        require(receipt['capture_sha256'] == digest(capture)
                and receipt['captured_bytes'] == receipt['observed_bytes'] == len(capture),
                'capture hash/length mismatch')
        require(report_binding['sha256'] == digest(raw) and report_binding['size_bytes'] == len(raw),
                'report hash/length mismatch')
        total += len(capture)
        report = json.loads(raw)
        canonical = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False)
        require(canonical not in canonical_reports, 'duplicate report across collection')
        canonical_reports.add(canonical)
        require(report['diagnostic_profiling'] is False, 'unchanged full harness required')
        groups[condition].append(raw)
        if condition.endswith('T'):
            bound = bind_trials(capture, raw, {
                'schema_version': 1, 'layout': 'codec-bench-b1',
                'trace_sha256': receipt['capture_sha256'], 'report_sha256': report_binding['sha256'],
                'source_sha256': sources[condition[0]], 'process_id': pid})
            joined.append({'ordinal': ordinal, 'condition': condition, 'binding': bound})
        else:
            require(not any(marker in capture for marker in (b'DRIVER ', b'RUNTIME ', b'KERNEL ')),
                    'trace records in plain capture')
        bindings.append({'ordinal': ordinal, 'condition': condition, 'pid': pid,
                         'receipt_sha256': digest(receipt_raw), 'capture_sha256': digest(capture),
                         'report_sha256': digest(raw)})
    require(type(result['captured_bytes']) is int and result['captured_bytes'] == total <= TOTAL_BYTES,
            'aggregate capture count mismatch')
    quality = {c: qualify(raw, sources[c[0]]) for c, raw in groups.items()}
    overhead = []
    for implementation in ('A', 'C'):
        for repeat, (plain, traced) in enumerate(zip(groups[implementation + 'P'],
                                                   groups[implementation + 'T']), 1):
            plain, traced = json.loads(plain), json.loads(traced)
            ps, ts = stats(plain['load_trials_ms']), stats(traced['load_trials_ms'])
            overhead.append({'implementation': implementation, 'occurrence_pair': repeat,
                             'batch': None, 'scope': 'checked_load',
                             'timings': {'load_trials_ms': {
                                 'plain': ps, 'traced': ts,
                                 'traced_over_plain_median': ts['median_ms'] / ps['median_ms'],
                                 'traced_over_plain_p95': ts['p95_ms'] / ps['p95_ms']}},
                             'first_load_ms': {'plain': plain['first_load_ms'], 'traced': traced['first_load_ms']},
                             'first_execution_ms': {'plain': plain['first_execution_ms'],
                                                    'traced': traced['first_execution_ms']}})
            for p, t in zip(plain['measurements'], traced['measurements']):
                require((p['batch'], p['scope']) == (t['batch'], t['scope']), 'overhead scope mismatch')
                timings = {}
                for field in ('trials_ms', 'cuda_stream_spans_ms'):
                    ps, ts = stats(p[field]), stats(t[field])
                    timings[field] = {'plain': ps, 'traced': ts,
                                      'traced_over_plain_median': ts['median_ms'] / ps['median_ms'],
                                      'traced_over_plain_p95': ts['p95_ms'] / ps['p95_ms']}
                overhead.append({'implementation': implementation, 'occurrence_pair': repeat,
                                 'batch': p['batch'], 'scope': p['scope'], 'timings': timings,
                                 'first_ms': {'plain': p['first_ms'], 'traced': t['first_ms']},
                                 'first_cuda_stream_span_ms': {'plain': p['first_cuda_stream_span_ms'],
                                                              'traced': t['first_cuda_stream_span_ms']}})
    return {'schema_version': 1, 'diagnostic_only': True, 'receipt_consistency_verified': True,
            'reported_codec_outputs_qualified': True, 'launch_provenance_verified': False,
            'performance_acceptance_evaluated': False, 'telemetry_coverage_verified': False,
            'plan_sha256': digest(plan_raw), 'sequence_result_sha256': digest(result_raw),
            'declared_sources': sources, 'jobs': bindings, 'qualification': quality,
            'traced_trials': joined, 'instrumentation_comparisons': overhead,
            'scope': 'All 12 supplied receipts/reports and six trace PID/ordinal joins. Receipt '
                     'consistency and distinct PIDs/hashes do not prove freshness, same-child file '
                     'production, executable/input/injection identity or collection supervision. '
                     'Instrumentation comparisons pair occurrence order within each condition, '
                     'not simultaneous controls or a balanced carryover experiment. No acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('sources', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.collection, json.loads(read_bounded(args.sources, REPORT_LIMIT)))
    result['tool_source_sha256'] = {
        name: digest(Path(__file__).with_name(name).read_bytes()) for name in (
            'analyze_readonly_trace.py', 'readonly_trace_sequence.py', 'bounded_child.py',
            'bind_cupti_trials.py', 'cupti_kernel_windows.py', 'cupti_trace_boundaries.py',
            'qualify_diagnostic_codec.py', 'phase2_summary.py')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
