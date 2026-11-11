"""Join source-assumed trace windows to declared benchmark report trial indices.

A hash declaration binds bytes, not launch provenance. Output qualification,
copy attribution and collection provenance remain separate, mandatory checks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from cupti_kernel_windows import summarize
from cupti_trace_boundaries import validate_boundaries

ROOT = Path(__file__).resolve().parents[2]
# Reviewed current harness layout; do not infer support from report self-description.
LAYOUT_SOURCES = {
    'crates/ncm-backend-cuda/examples/codec_bench.rs':
        '2648faa43e439efd82a0a318962dec6171d1a808e1fc4398f0496775f0b6e11a',
    'crates/ncm-backend-cuda/examples/support/stages.rs':
        'd716c58d5a9432615821a7a87509aee4210e2785d767a1be8c25eaec511339b5',
}
SCOPES = ('encode', 'decode', 'reconstruct', 'resident_decoder_block')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timing(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'invalid recorded timing')
    return value


def bind_trials(trace_raw, report_raw, declaration):
    """Declaration must come from reviewed evidence; this function cannot prove it."""
    require(set(declaration) == {'schema_version', 'layout', 'trace_sha256', 'report_sha256',
                                'source_sha256', 'process_id'}, 'invalid binding declaration fields')
    require(type(declaration['schema_version']) is int and declaration['schema_version'] == 1,
            'unsupported binding declaration')
    layout = declaration['layout']
    require(layout in ('codec-bench-b1', 'historical-block4'), 'unsupported trace layout')
    require(type(declaration['process_id']) is int and declaration['process_id'] > 0,
            'invalid declared process identity')
    sources = declaration['source_sha256']
    require(isinstance(sources, dict) and bool(sources), 'missing declared sources')
    require(all(isinstance(p, str) and isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h)
                for p, h in sources.items()), 'invalid declared source digests')
    require(digest(trace_raw) == declaration['trace_sha256'], 'trace hash mismatch')
    require(digest(report_raw) == declaration['report_sha256'], 'report hash mismatch')
    report = json.loads(report_raw)
    require(report['source_sha256'] == sources, 'report source inventory mismatch')
    require(all(sources.get(p) == h for p, h in LAYOUT_SOURCES.items()),
            'unreviewed benchmark source layout')
    require(type(report['schema_version']) is int and report['schema_version'] == 1,
            'unsupported report schema')
    require(report['runtime'] == 'rust-candle' and report['device'] == 'cuda'
            and report['precision'] == 'fp32' and report['index_dtype'] == 'u32',
            'unsupported report runtime/precision')
    require(report['warmups'] == 5 and report['trials'] == 30, 'unsupported trial protocol')
    require(report['diagnostic_profiling'] is (layout == 'historical-block4'),
            'report profiling mode does not match layout')
    require(report['workload_sha256'] == digest((ROOT / 'tests/fixtures/codec-bench-v1.json').read_bytes()),
            'workload hash mismatch')
    require(report['manifest_sha256'] == digest(
        (ROOT / 'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json').read_bytes()),
        'artifact manifest hash mismatch')
    expected_order = [(b, s) for b in (1, 2, 4) for s in SCOPES]
    require([(r['batch'], r['scope']) for r in report['measurements']] == expected_order,
            'incomplete or reordered report scopes')
    for row in report['measurements']:
        for field in ('trials_ms', 'cuda_stream_spans_ms'):
            require(isinstance(row[field], list) and len(row[field]) == 30, 'incomplete trial array')
            for value in row[field]:
                timing(value)
        timing(row['first_ms'])
        timing(row['first_cuda_stream_span_ms'])
    boundaries = validate_boundaries(trace_raw, layout)
    require(boundaries['process_id'] == declaration['process_id'], 'trace process identity mismatch')
    result = summarize(trace_raw, layout)
    batch = 1 if layout == 'codec-bench-b1' else 4
    row = next(r for r in report['measurements']
               if (r['batch'], r['scope']) == (batch, 'resident_decoder_block'))
    require(len(result['windows']) == 36, 'incomplete block windows')
    for window in result['windows']:
        work = window['assumed_work']
        if work['phase'] == 'first':
            recorded = {'wall_ms': row['first_ms'], 'cuda_stream_span_ms': row['first_cuda_stream_span_ms']}
        elif work['phase'] == 'warmup':
            # The unchanged harness runs warmups but does not serialize their timings.
            recorded = None
        elif work['phase'] == 'trial':
            index = work['index']
            recorded = {'wall_ms': row['trials_ms'][index],
                        'cuda_stream_span_ms': row['cuda_stream_spans_ms'][index]}
        else:
            raise ValueError('unexpected block phase')
        window['recorded_timing'] = recorded
    result.update({
        'diagnostic_only': True, 'declared_hash_binding_verified': True,
        'launch_provenance_verified': False, 'benchmark_outputs_qualified': False,
        'report_sha256': digest(report_raw), 'declared_process_id': declaration['process_id'],
        'declared_source_sha256': sources,
        'join_scope': 'Source-derived ordinal join, not nearest-duration matching. Warmup timings '
                      'are unavailable, not zero. Hash declarations do not independently prove '
                      'same-child collection, binary/source provenance or benchmark output quality.',
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'report', 'declaration', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    declaration_raw = args.declaration.read_bytes()
    result = bind_trials(args.trace.read_bytes(), args.report.read_bytes(), json.loads(declaration_raw))
    result['declaration_sha256'] = digest(declaration_raw)
    result['source_sha256'] = {
        name: digest(Path(__file__).with_name(name).read_bytes())
        for name in ('bind_cupti_trials.py', 'cupti_kernel_windows.py', 'cupti_trace_boundaries.py')}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
