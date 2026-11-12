"""Independent offline clock process/receipt consistency, not full qualification.

No file IO or collection. Run after supervision/telemetry; this does not authorize
frozen rehash, establish restoration, authenticate producers or qualify outputs.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from clock_library_contract import signature
from clock_process_observation import ORDER, RUNTIMES
from clock_codec_sequence import CAPTURE_BYTES, REPORT_BYTES, WORK_SECONDS
from observe_readonly_process import MAX_SAMPLES, MAP_INVENTORY_BYTES, SAMPLE_INTERVAL_NS
from readonly_trace_preflight import require
from validate_readonly_lifecycle import Clocks, contains, same_process, receipt_complete, digest
from verify_readonly_processes import number, key, mapping_key


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid evidence digest')


def validate_processes(jobs, result, contract, *, injection, work_deadline):
    """Check all 18 records against separately frozen commands and library declarations.

    Required mappings must appear in workload samples. Known optional paths must
    match even on first appearance, but need not appear. Unknown mappings remain
    explicitly post-only; current file hashes belong in a later finalization gate.
    """
    require(type(work_deadline) in (int, float) and math.isfinite(work_deadline) and work_deadline > 0,
            'finite original work deadline required')
    require(type(result['schema_version']) is int and result['schema_version'] == 1
            and result['status'] == 'complete' and result['stop_reason'] is None
            and result['external_cancelled'] is False and result['diagnostic_only'] is True
            and result['python_copies_deferred'] is True
            and type(result['work_deadline_monotonic_seconds']) in (int, float)
            and result['work_deadline_monotonic_seconds'] == work_deadline,
            'incomplete clock sequence or mismatched original deadline')
    require(all(result[k] is False for k in ('launch_provenance_verified', 'benchmark_outputs_qualified',
                                           'whole_run_supervision_provided')), 'unsupported sequence claim')
    require(type(contract['schema_version']) is int and contract['schema_version'] == 1
            and contract['prepared_declaration_consistency_verified'] is True
            and contract['runtime_mapping_verified'] is False and contract['launch_ready'] is False,
            'unbound or unsupported library contract')
    require(isinstance(jobs, list) and isinstance(result['jobs'], list)
            and len(jobs) == len(result['jobs']) == 18, 'all eighteen jobs required')
    require(set(contract['workloads']) == set(contract['libraries']) == set(RUNTIMES), 'incomplete runtime declarations')
    signature(injection); signature(contract['taskset'])
    known = {}
    require(isinstance(contract['known_libraries'], list) and 0 < len(contract['known_libraries']) <= 256,
            'bounded known inventory required')
    for value in contract['known_libraries']:
        signature(value)
        require(value['path'] not in known, 'duplicate known library path')
        known[value['path']] = value
    require(sum(v['bytes'] for v in known.values()) <= 8*1024**3, 'known inventory exceeds byte bound')
    for runtime in RUNTIMES:
        signature(contract['workloads'][runtime]['image'])
        required = contract['libraries'][runtime]
        require(isinstance(required, list) and required, 'missing mandatory library inventory')
        for value in required:
            require(value['path'] in known and signature(value) == signature(known[value['path']]),
                    'required/known declarations differ')
    prepared = dict(known)
    for value in [injection, contract['taskset'], *(v['image'] for v in contract['workloads'].values())]:
        require(value['path'] not in prepared or signature(prepared[value['path']]) == signature(value),
                'prepared image/library declarations conflict')
        prepared[value['path']] = value
    clocks = Clocks()
    sequence = clocks.lifetime(result)
    require(sequence['end'][1] / 1e9 <= work_deadline, 'sequence finished beyond original work deadline')
    files, pids, reports, bindings, previous_end = {}, set(), set(), [], None
    for ordinal, (job, row, runtime) in enumerate(zip(jobs, result['jobs'], ORDER), 1):
        require(number(row['ordinal'], 1) == ordinal and row['runtime'] == job['runtime'] == runtime
                and number(row['repeat'], 1) == number(job['repeat'], 1) == (ordinal-1)//3+1
                and job['workload'] == 'codec', 'clock job order/repetition mismatch')
        expected = contract['workloads'][runtime]
        argv = job['argv']
        require(isinstance(argv, list) and len(argv) >= 5
                and all(isinstance(a, str) and a and '\0' not in a for a in argv)
                and argv[:4] == [contract['taskset']['path'], '-c', '2', expected['command_path']]
                and argv[-1] == job['report']
                and all(isinstance(p, str) and '\0' not in p and Path(p).is_absolute()
                        and '..' not in Path(p).parts for p in (expected['command_path'], job['report'], job['cwd'])),
                'command/image/CPU/report declaration mismatch')
        require(job['report'] not in reports, 'duplicate planned report path')
        reports.add(job['report'])
        receipt, observed = row['receipt'], row['observation']
        pid = receipt_complete(receipt)
        require(pid not in pids and pid not in (sequence['pid'], sequence['parent'])
                and number(observed['pid'], 1) == pid, 'duplicate/mismatched process PID')
        pids.add(pid)
        require(receipt['argv'] == argv and receipt['cwd'] == job['cwd']
                and receipt['process_observer_enabled'] is True
                and receipt['output_limit_bytes'] == CAPTURE_BYTES and receipt['report_limit_bytes'] == REPORT_BYTES
                and receipt['timeout_seconds'] == WORK_SECONDS
                and type(receipt['termination_grace_seconds']) in (int, float) and receipt['termination_grace_seconds'] == 0
                and receipt['termination_grace_exhausted'] is False and receipt['termination_signal'] == 'not_attempted'
                and receipt['workload_output_qualified'] is False and receipt['executable_identity_verified'] is False,
                'receipt command/limits/scope mismatch')
        require(number(receipt['captured_bytes']) == number(receipt['observed_bytes']) <= CAPTURE_BYTES,
                'capture byte accounting mismatch')
        sha(receipt['capture_sha256'])
        report = receipt['report_capture']
        require(report['path'] == job['report'] and 0 < number(report['size_bytes']) <= REPORT_BYTES
                and report['post_exit_hash_verified'] is True and report['producer_identity_verified'] is False,
                'missing or mismatched report receipt')
        sha(report['sha256']); key(report)
        child = clocks.lifetime(receipt)
        contains(sequence, child); same_process(sequence, child)
        require(previous_end is None or previous_end <= child['start'][0], 'overlapping child receipt envelopes')
        previous_end = child['end'][1]
        spawn = receipt['spawn_returned_monotonic_seconds']
        require(type(spawn) in (int, float) and math.isfinite(spawn)
                and child['start'][0]/1e9 <= spawn <= child['end'][1]/1e9 <= work_deadline,
                'invalid/late owned spawn evidence')
        require(type(observed['schema_version']) is int and observed['schema_version'] == 1
                and observed['status'] == 'complete' and observed['issues'] == []
                and observed['observation_failures'] == [] and number(observed['process_start_ticks'], 1),
                'incomplete process observations')
        require(all(observed[k] is False for k in ('launch_provenance_verified','whole_lifetime_mapping_coverage',
                                                  'mapped_file_bytes_verified','environment_values_serialized')),
                'unsupported stronger observation claim')
        transient = number(observed['transient_samples_discarded'])
        require(sum(number(observed[k]) for k in ('startup_incomplete_samples','proc_unavailable_samples',
                                                 'startup_permission_denied_samples')) <= transient,
                'transient counters inconsistent')
        samples, maps = observed['samples'], observed['mapping_sets']
        require(isinstance(samples, list) and samples and len(samples)+transient <= MAX_SAMPLES
                and isinstance(maps, dict) and maps, 'missing/oversized mapping observations')
        require(len(json.dumps(observed, indent=2, allow_nan=False).encode())+1 <= 16*1024**2,
                'observation record exceeds bound')
        encoded_bytes, used, work_keys, prior_start, prior_end = 0, set(), set(), None, None
        for map_hash, mapping in maps.items():
            require(map_hash == digest(mapping) and set(mapping) == {'files','non_file_executable_mappings'}
                    and isinstance(mapping['files'], list) and mapping['files'], 'invalid mapping digest/fields')
            encoded_bytes += len(json.dumps(mapping, sort_keys=True).encode())
            special = mapping['non_file_executable_mappings']
            require(isinstance(special, list) and all(isinstance(v, str) for v in special)
                    and special == sorted(set(special)), 'invalid non-file mapping labels')
            paths = []
            for value in mapping['files']:
                require(set(value) == {'path','device','inode','path_device','path_inode','kernel_map_link_verified'},
                        'invalid executable mapping fields')
                path = value['path']
                require(isinstance(path, str) and Path(path).is_absolute() and '..' not in Path(path).parts
                        and '\0' not in path and '\\' not in path and not path.endswith(' (deleted)'),
                        'unsupported mapped path')
                key(value); actual = mapping_key(value)
                require(value['inode'] == value['path_inode'] and value['kernel_map_link_verified'] is True,
                        'missing kernel/path inode binding')
                require(actual != key(injection) and path != injection['path'], 'injection identity/path in plain process')
                require(path not in prepared or actual == key(prepared[path]), 'known candidate/image differs from preparation')
                require(path not in files or files[path] == value, 'mapped identity changed across jobs')
                files[path] = value
                paths.append(path)
            require(paths == sorted(set(paths)), 'unordered/duplicate mapped paths')
        require(encoded_bytes <= MAP_INVENTORY_BYTES, 'mapping inventory exceeds bound')
        for sample in samples:
            start, end = number(sample['monotonic_start_ns']), number(sample['monotonic_end_ns'])
            require(child['start'][1] <= start <= end <= child['end'][0]
                    and (prior_start is None or start-prior_start >= SAMPLE_INTERVAL_NS)
                    and (prior_end is None or prior_end <= start), 'invalid sample timing/envelope')
            prior_start, prior_end = start, end
            phase = sample['phase']
            require(phase in ('launcher','workload') and sample['injection_observed'] is False,
                    'unsupported phase or injection claim')
            affinity = sample['affinity']
            require(isinstance(affinity, list) and affinity
                    and all(type(v) is int and v >= 0 for v in affinity) and affinity == sorted(set(affinity)),
                    'invalid sampled affinity')
            mapping = maps[sample['mapping_sha256']]
            used.add(sample['mapping_sha256'])
            keys = {mapping_key(v) for v in mapping['files']}
            require(key(expected['image'] if phase == 'workload' else contract['taskset']) in keys,
                    'declared image not observed')
            if phase == 'workload':
                require(affinity == [2], 'workload CPU differs')
                work_keys.update(keys)
        require(used == maps.keys() and work_keys, 'unused maps or no complete workload sample')
        require({key(v) for v in contract['libraries'][runtime]} <= work_keys, 'mandatory libraries not observed')
        bindings.append({'ordinal':ordinal,'runtime':runtime,'pid':pid,'samples':len(samples),
                         'receipt_canonical_sha256':digest(receipt),'observation_canonical_sha256':digest(observed)})
    return {'schema_version':1, 'process_receipt_consistency_verified':True, 'jobs':bindings,
            'observed_files':deepcopy(files), 'known_paths_observed':sorted(files.keys() & known.keys()),
            'prepared_paths_observed':sorted(files.keys() & prepared.keys()),
            'unlisted_paths_observed':sorted(files.keys() - prepared.keys()), 'clock_domain_offset_ns':clocks.offset(),
            'launch_provenance_verified':False, 'mapped_file_bytes_verified':False,
            'restoration_verified':False, 'telemetry_coverage_verified':False,
            'benchmark_outputs_qualified':False, 'frozen_files_rechecked':False,
            'scope':'Supplied eighteen-job receipt/sample/declaration consistency only. Not producer authentication, '
                    'complete mapping coverage, clock restoration, telemetry coverage, output quality or file postflight.'}
