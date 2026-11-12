"""Offline full CPU/GPU contents and clock-diagnostic receipt coverage.

Reuse the unchanged 901-sample CPU and GPU unit/timezone parsers. No collection,
policy commands, package execution or frozen-file rehash. ZoneInfo may read timezone
data; raw-input hashes are not a claim about immutable files or producer identity.
"""
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
from zoneinfo import ZoneInfo

from clock_process_observation import ORDER
from readonly_trace_preflight import require
from validate_readonly_lifecycle import Clocks, digest, receipt_complete
from validate_readonly_telemetry import cpu_records, gpu_records, utc_ns, integer
from verify_readonly_processes import key


def validate(cpu_raw, gpu_raw, sequence, restorer_receipt, telemetry, ready, *,
             expected_cpu_source, gpu_uuid, timezone_name, cpu_path, cpu_argv):
    """Bind raw telemetry/readiness and bracket the restorer plus eighteen jobs.

    Caller supplies launch-bound source/UUID/IANA zone/CPU path/argv, not values
    inferred from the collection to make it pass. Full lifecycle, process maps,
    command authenticity, policy enforcement/restoration and output quality remain
    separate. One-second samples bracket windows, not continuous/per-call clocks.
    """
    cpu = cpu_records(cpu_raw,expected_cpu_source)
    gpu = gpu_records(gpu_raw,gpu_uuid,timezone_name)
    cpu_hash,gpu_hash = (hashlib.sha256(raw).hexdigest() for raw in (cpu_raw,gpu_raw))
    require(isinstance(cpu_path,str) and Path(cpu_path).is_absolute() and '..' not in Path(cpu_path).parts
            and '\0' not in cpu_path and isinstance(cpu_argv,list) and cpu_argv
            and all(isinstance(v,str) and v and '\0' not in v for v in cpu_argv)
            and Path(cpu_argv[0]).is_absolute(), 'explicit CPU command/path required')
    require(type(telemetry['schema_version']) is int and telemetry['schema_version'] == 1
            and telemetry['status'] == 'complete' and telemetry['ready'] is True
            and telemetry['errors'] == [] and telemetry['interrupt_signal'] is None
            and all(telemetry[k] is False for k in ('telemetry_contents_qualified','sequence_coverage_verified',
                                                  'launch_provenance_verified')), 'incomplete telemetry lifecycle record')
    cpu_receipt = telemetry['cpu_receipt']
    receipt_complete(cpu_receipt)
    report = cpu_receipt['report_capture']
    require(cpu_receipt['argv'] == cpu_argv and report['path'] == cpu_path
            and integer(report['size_bytes'],1) == len(cpu_raw) and report['sha256'] == cpu_hash
            and report['post_exit_hash_verified'] is True and report['producer_identity_verified'] is False
            and telemetry['gpu_sha256'] == gpu_hash, 'raw telemetry receipt binding differs')
    key(report)
    require(type(ready['schema_version']) is int and ready['schema_version'] == 1
            and ready['kind'] == 'telemetry_ready'
            and integer(ready['worker_pid'],1) == integer(telemetry['lifetime']['pid'],1)
            and integer(ready['gpu_pid'],1) == integer(telemetry['gpu_pid'],1)
            and ready['observer_affinity'] == [0] and all(type(v) is int for v in ready['observer_affinity']),
            'readiness identity/affinity differs')
    require(digest(ready['cpu_metadata']) == digest(cpu['metadata']), 'readiness CPU metadata differs from raw contents')
    raw_rows = list(csv.reader(io.StringIO(gpu_raw.decode('utf-8')),skipinitialspace=True,strict=True))
    require(ready['first_gpu_row'] == raw_rows[1], 'readiness GPU row differs from raw contents')
    local = datetime.fromisoformat(ready['local_time_with_offset'])
    require(local.utcoffset() is not None and local.utcoffset() == local.astimezone(ZoneInfo(timezone_name)).utcoffset()
            and max(cpu['utc_ns'][0],gpu[0]['utc_ns']) <= utc_ns(ready['local_time_with_offset'])
            <= utc_ns(ready['record_clock']['utc']),
            'readiness local time/zone differs')
    clocks = Clocks()
    ready_clock = clocks.stamp(ready['record_clock'])
    cpu_life = clocks.lifetime(cpu_receipt)
    sample_clocks = [clocks.cpu_sample(row) for row in cpu['samples']]
    require(cpu_life['start'][1] <= sample_clocks[0][0] and sample_clocks[-1][1] <= cpu_life['end'][0]
            and sample_clocks[0][1] <= ready_clock[0] <= ready_clock[1] <= cpu_life['end'][0],
            'readiness/CPU samples outside CPU receipt')
    require(type(sequence['schema_version']) is int and sequence['schema_version'] == 1
            and sequence['status'] == 'complete' and sequence['stop_reason'] is None
            and sequence['external_cancelled'] is False and sequence['diagnostic_only'] is True
            and isinstance(sequence['jobs'],list) and len(sequence['jobs']) == 18, 'complete eighteen-job sequence required')
    pids = set()
    def window(receipt):
        pid = receipt_complete(receipt)
        require(pid not in pids, 'duplicate telemetry process window')
        pids.add(pid)
        life = clocks.lifetime(receipt)
        start,end = utc_ns(receipt['started_utc']),utc_ns(receipt['finished_utc'])
        require(start < end and cpu['utc_ns'][0] <= start < end <= cpu['utc_ns'][-1]
                and gpu[0]['utc_ns'] <= start < end <= gpu[-1]['utc_ns'], 'telemetry does not bracket receipt')
        return {'pid':pid,'started_utc':receipt['started_utc'],'finished_utc':receipt['finished_utc'],
                'receipt_canonical_sha256':digest(receipt),
                'cpu_interval_end_indices':[i for i in range(1,len(cpu['utc_ns']))
                    if cpu['utc_ns'][i-1] < end and cpu['utc_ns'][i] > start],
                'gpu_interval_end_indices':[i for i in range(1,len(gpu))
                    if gpu[i-1]['utc_ns'] < end and gpu[i]['utc_ns'] > start],
                'gpu_sample_indices_inside':[r['index'] for r in gpu if start <= r['utc_ns'] <= end]},life
    outer,outer_life = window(restorer_receipt)
    require(ready_clock[1] <= outer_life['start'][0], 'restorer precedes readiness')
    jobs,previous_end = [],None
    for ordinal,(runtime,row) in enumerate(zip(ORDER,sequence['jobs']),1):
        require(integer(row['ordinal'],1) == ordinal and row['runtime'] == runtime
                and integer(row['repeat'],1) == (ordinal-1)//3+1, 'clock receipt order/repetition differs')
        current,life = window(row['receipt'])
        start,end = utc_ns(current['started_utc']),utc_ns(current['finished_utc'])
        require(utc_ns(outer['started_utc']) <= start < end <= utc_ns(outer['finished_utc'])
                and outer_life['start'][1] <= life['start'][0] <= life['end'][1] <= outer_life['end'][0]
                and (previous_end is None or previous_end <= start), 'job outside restorer or out of order')
        previous_end = end
        jobs.append(dict(current,ordinal=ordinal,runtime=runtime,repeat=row['repeat']))
    return {'schema_version':1,'diagnostic_only':True,'telemetry_contents_validated':True,
            'readiness_contents_bound':True,'receipt_windows_bracketed':True,
            'cpu_sha256':cpu_hash,'gpu_sha256':gpu_hash,'declared_cpu_source_sha256':expected_cpu_source,
            'declared_gpu_uuid':gpu_uuid,'gpu_timezone':timezone_name,
            'records_canonical_sha256':{name:digest(value) for name,value in (
                ('sequence',sequence),('restorer_receipt',restorer_receipt),('telemetry',telemetry),('ready',ready),
                ('cpu_samples',cpu['samples']))},
            'clock_domain_offset_ns':clocks.offset(),'cpu_metadata':cpu['metadata'],'cpu_completion':cpu['completion'],
            'missing_cpu_optional':cpu['missing_optional'],'unavailable_cpu_activity':cpu['unavailable_activity'],
            'cpu_lateness_seconds':[r['lateness_seconds'] for r in cpu['samples']],
            'cpu_read_seconds':[(r['monotonic_end_ns']-r['monotonic_start_ns'])/1e9 for r in cpu['samples']],
            'cpu_sample_gaps_seconds':[(b-a)/1e9 for a,b in zip(cpu['utc_ns'],cpu['utc_ns'][1:])],
            'gpu_samples':gpu,'gpu_sample_gaps_seconds':[(b['utc_ns']-a['utc_ns'])/1e9 for a,b in zip(gpu,gpu[1:])],
            'gpu_first_relative_to_cpu_first_seconds':(gpu[0]['utc_ns']-cpu['utc_ns'][0])/1e9,
            'gpu_last_relative_to_cpu_last_seconds':(gpu[-1]['utc_ns']-cpu['utc_ns'][-1])/1e9,
            'restorer':outer,'jobs':jobs,'launch_provenance_verified':False,
            'continuous_gpu_observation_verified':False,'clock_enforcement_verified':False,
            'hardware_policy_restoration_verified':False,'performance_acceptance_evaluated':False,
            'benchmark_outputs_qualified':False,'postflight_call_order_enforced':False,
            'scope':'Supplied full telemetry contents, readiness/raw receipt binding and coarse restorer/job '
                    'bracketing only. All gaps/unavailable values retained. Not quietness, continuous clocks, '
                    'hardware enforcement/restoration, live provenance, output quality or finalization ordering.'}
