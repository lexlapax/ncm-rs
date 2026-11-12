"""Offline full-report clock diagnostic qualification, not performance acceptance.

Reads the existing frozen reference archive and codec fixture/manifest through the
unchanged qualifier. No workloads, Torch imports or final frozen-file rehash. Raw
report bytes/receipts are supplied by the caller; file/producer binding is separate.
"""
import hashlib
import json
import math
import re

from clock_process_observation import ORDER
from clock_codec_sequence import REPORT_BYTES
from prepare_clock_python import FIELDS
from qualify_diagnostic_codec import (ROOT, NATIVE_SOURCES, LAYOUT_SOURCES,
                                      frozen_references, validate_group)
from phase2_summary import codec, require
from validate_readonly_lifecycle import digest, receipt_complete
from validate_readonly_telemetry import unique_object, reject_constant
from verify_readonly_processes import number, key


def numeric(value, positive=False):
    require(type(value) in (int,float) and math.isfinite(value) and (not positive or value > 0),
            'invalid numeric report value')


def report_numbers(report):
    """Do not let Boolean JSON values stand in for reported numbers."""
    for name in ('first_load_ms','first_execution_ms'): numeric(report[name],True)
    arrays = [report['load_trials_ms']]
    for row in report['measurements']:
        number(row['batch'],1); number(row['output_ids_count']); number(row['output_float_elements'])
        numeric(row['first_ms'],True)
        arrays.extend((row['trials_ms'],row['cuda_stream_spans_ms']))
        if row['first_logits'] is not None:
            require(isinstance(row['first_logits'],list), 'logit array required')
            for value in row['first_logits']: numeric(value)
    for row in report['instrumented_stages']:
        number(row['batch'],1)
        arrays.extend(row['cuda_stream_spans_ms'].values())
    for array in arrays:
        require(isinstance(array,list) and len(array) == 30, 'thirty reported timing trials required')
        for value in array: numeric(value,True)


def qualify(raw_reports, jobs, *, sources, python_metadata):
    """Qualify six reports/runtime in fixed order; no prefix, retry or copied process.

    The codec checker expects triples. Partition deterministically into repeats
    1–3 and 4–6, checking both native runtimes and all current Python reports directly
    against the same frozen reference. No selection/aggregation into acceptance.
    """
    require(isinstance(raw_reports,list) and isinstance(jobs,list) and len(raw_reports) == len(jobs) == 18,
            'eighteen raw reports and receipts required')
    require(set(sources) == {'previous','candidate'}, 'both native source declarations required')
    for expected in sources.values():
        require(isinstance(expected,dict) and set(expected) == NATIVE_SOURCES
                and all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for h in expected.values())
                and all(expected[p] == h for p,h in LAYOUT_SOURCES.items()), 'unreviewed native source inventory/layout')
    require(isinstance(python_metadata,dict) and set(python_metadata) == set(FIELDS),
            'complete historical Python metadata declaration required')
    groups = {runtime:[] for runtime in ('previous','python','candidate')}
    bindings,pids,paths = [],set(),set()
    for ordinal,(runtime,raw,job) in enumerate(zip(ORDER,raw_reports,jobs),1):
        require(number(job['ordinal'],1) == ordinal and job['runtime'] == runtime
                and number(job['repeat'],1) == (ordinal-1)//3+1, 'clock report order/repetition mismatch')
        require(isinstance(raw,bytes) and 0 < len(raw) <= REPORT_BYTES, 'empty or oversized report bytes')
        receipt = job['receipt']; pid = receipt_complete(receipt); capture = receipt['report_capture']
        require(pid not in pids and capture['path'] not in paths, 'duplicate process/report path')
        pids.add(pid); paths.add(capture['path']); key(capture)
        sha = hashlib.sha256(raw).hexdigest()
        require(number(capture['size_bytes'],1) == len(raw) and capture['sha256'] == sha
                and capture['post_exit_hash_verified'] is True and capture['producer_identity_verified'] is False,
                'report raw receipt mismatch')
        report = json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
        require(type(report['schema_version']) is int and report['schema_version'] == 1
                and type(report['warmups']) is int and report['warmups'] == 5
                and type(report['trials']) is int and report['trials'] == 30
                and report.get('diagnostic_profiling',False) is False, 'invalid clock report protocol/profiling')
        report_numbers(report)
        if runtime == 'python':
            require(digest({k:report[k] for k in FIELDS}) == digest(python_metadata),
                    'current Python report metadata differs from declared reference')
        groups[runtime].append(report)
        bindings.append({'ordinal':ordinal,'runtime':runtime,'repeat':job['repeat'],'pid':pid,
                         'report_sha256':sha,'report_canonical_sha256':digest(report),
                         'receipt_canonical_sha256':digest(receipt)})
    for reports in groups.values():
        require(len(reports) == 6 and len({digest(r) for r in reports}) == 6,
                'six distinct reports per runtime required')
    reference,reference_binding = frozen_references()
    require(all(digest({k:r[k] for k in FIELDS}) == digest(python_metadata) for r in reference),
            'declared Python metadata differs from frozen reference')
    validate_group(reference,'pytorch-pinned-eager-snapshot-fix',python_metadata['source_sha256'])
    qualified = []
    for start in (0,3):
        current_python = groups['python'][start:start+3]
        validate_group(current_python,'pytorch-pinned-eager-snapshot-fix',python_metadata['source_sha256'])
        for runtime in ('previous','candidate'):
            native = groups[runtime][start:start+3]
            validate_group(native,'rust-candle',sources[runtime])
            runs = {'rust':native,'python':reference}
            if runtime == 'previous':
                # A third group uses Python allocator metadata but compares its
                # outputs directly to the original frozen Python baseline, not
                # transitively via the native candidate or new Python report zero.
                runs['current_python'] = current_python
            result = codec(ROOT,'cuda',runs=runs)
            qualified.append({'native_runtime':runtime,'repeats':list(range(start+1,start+4)),
                              'includes_current_python':runtime == 'previous',
                              'quality':[{k:row[k] for k in ('batch','scope','quality','output_ids_sha256')}
                                         for row in result['measurements']]})
    return {'schema_version':1,'diagnostic_only':True,'reported_codec_outputs_qualified':True,
            'jobs':bindings,'groups':qualified,'frozen_reference_binding':reference_binding,
            'declared_native_source_sha256':sources,'declared_python_metadata':python_metadata,
            'launch_provenance_verified':False,'independent_processes_verified':False,
            'capture_files_verified':False,'performance_acceptance_evaluated':False,
            'operator_trace_parity_verified':False,'postflight_call_order_enforced':False,
            'scope':'All eighteen supplied full reports: B1/B2/B4, twelve scopes, IDs/retained counts, '
                    'first decode/reconstruct logits and timing/stage/load structure against frozen Python. '
                    'Unchanged FP32 tolerances. Not file/producer freshness, unreported floating outputs, '
                    '642-trace parity, runtime library equivalence or performance acceptance.'}
