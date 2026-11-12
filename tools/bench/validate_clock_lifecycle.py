"""Offline sibling-supervisor lifecycle consistency, not launch ancestry enforcement.

The caller must actually reap its coordinator and both supervisor branches before
finalization. Records cannot prove that ownership, complete telemetry or outputs.
"""
import signal

from clock_process_observation import ORDER
from readonly_trace_preflight import require
from validate_clock_restoration import validate as validate_restoration, finite
from validate_readonly_lifecycle import Clocks, contains, same_process, receipt_complete, digest
from verify_readonly_processes import number
from verify_clock_processes import sha


def validate(collection, restorer_receipt, restoration, obligation, sequence,
             observer_supervisor, telemetry, ready, cpu_samples, *, commands, cwd,
             work_deadline, external_cancel_check_enabled):
    """Bind both sibling branches and all 18 job envelopes to one clock domain.

    Readiness/CPU/GPU *contents*, role argv/environment/images, raw captures and
    per-job observations/outputs remain separate gates. This does not authorize
    frozen rehash or make a generic adoption-and-KILL ancestor safe for restoration.
    """
    transaction = validate_restoration(restoration, obligation, sequence, commands=commands,
        cwd=cwd, work_deadline=work_deadline, external_cancel_check_enabled=external_cancel_check_enabled)
    clocks = Clocks()
    coord = clocks.lifetime(collection)
    outer = clocks.lifetime(restorer_receipt)
    rest = clocks.lifetime({'schema_version':1, 'lifetime':restoration['lifetime']})
    observer = clocks.lifetime(observer_supervisor)
    worker_receipt = observer_supervisor['worker']
    worker = clocks.lifetime(worker_receipt)
    tel = clocks.lifetime(telemetry)
    cpu = clocks.lifetime(telemetry['cpu_receipt'])
    seq = clocks.lifetime(sequence)
    require(receipt_complete(restorer_receipt) == rest['pid'] and rest['parent'] == coord['pid'],
            'restorer is not a captured coordinator child')
    same_process(coord, outer); contains(coord, outer); contains(outer, rest)
    require(observer['parent'] == coord['pid'] and number(collection['observer_pid'],1) == observer['pid'],
            'observer supervisor is not a sibling of restorer')
    contains(coord, observer); contains(observer, worker)
    require(receipt_complete(worker_receipt) == tel['pid'] and tel['parent'] == observer['pid'],
            'telemetry is not observer supervisor child')
    same_process(observer, worker); contains(worker, tel)
    same_process(tel, cpu); contains(tel, cpu)
    cpu_pid = receipt_complete(telemetry['cpu_receipt'])
    for receipt,envelope in ((worker_receipt,worker),(telemetry['cpu_receipt'],cpu)):
        require(finite(receipt['termination_grace_seconds']) == 0
                and receipt['termination_signal'] == 'not_attempted'
                and receipt['termination_grace_exhausted'] is False
                and receipt['process_observer_enabled'] is False
                and receipt['workload_output_qualified'] is False and receipt['executable_identity_verified'] is False
                and number(receipt['captured_bytes']) == number(receipt['observed_bytes'])
                <= number(receipt['output_limit_bytes'],1), 'observer capture accounting/termination differs')
        sha(receipt['capture_sha256'])
        require(envelope['start'][0]/1e9 <= finite(receipt['spawn_returned_monotonic_seconds'])
                <= envelope['end'][1]/1e9, 'observer spawn outside receipt')
    gpu_pid = number(telemetry['gpu_pid'],1)
    pids = [coord['pid'],coord['parent'],rest['pid'],observer['pid'],tel['pid'],cpu_pid,gpu_pid,
            transaction['set_pid'],transaction['worker_pid'],transaction['reset_pid']]
    require(len(set(pids)) == len(pids), 'duplicate collector/observer/transaction PIDs')
    require(collection['status'] == 'complete' and collection['errors'] == []
            and collection['interrupt_signal'] is None and 'sequence_status' not in collection
            and collection['restoration_supervisor_status'] == collection['telemetry_status'] == 'complete',
            'incomplete clock coordinator')
    require(all(collection[k] is False for k in ('launch_provenance_verified','telemetry_contents_qualified',
            'sequence_coverage_verified','benchmark_outputs_qualified')), 'unsupported coordinator claim')
    require(observer_supervisor['status'] == 'complete' and observer_supervisor['errors'] == []
            and observer_supervisor['parent_lost'] is False and observer_supervisor['orphan_observed'] is False
            and observer_supervisor['subreaper_restored'] is True and observer_supervisor['interrupt_signal'] is None
            and observer_supervisor['launch_provenance_verified'] is False
            and observer_supervisor['benchmark_outputs_qualified'] is False, 'observer supervisor incomplete')
    cleanup = observer_supervisor['cleanup']
    require(cleanup['complete'] is True and all(cleanup[k] == [] for k in
            ('signalled','reaped','remaining','errors')), 'observer orphan cleanup is not success')
    shutdown = collection['observer_shutdown']
    require(shutdown['verified'] is True and type(shutdown['returncode']) is int
            and shutdown['returncode'] == 0 and shutdown['kill_escalation'] is False,
            'observer supervisor not cleanly reaped')
    require(telemetry['status'] == 'complete' and telemetry['ready'] is True and telemetry['errors'] == []
            and telemetry['interrupt_signal'] is None
            and all(telemetry[k] is False for k in ('telemetry_contents_qualified','sequence_coverage_verified',
                                                  'launch_provenance_verified')), 'incomplete telemetry worker')
    shutdown = telemetry['gpu_shutdown']
    require(shutdown['verified'] is True and type(shutdown['returncode']) is int
            and shutdown['returncode'] in (0,-signal.SIGTERM) and shutdown['kill_escalation'] is False,
            'GPU observer not cleanly stopped')
    require(finite(restorer_receipt['timeout_seconds']) == 660
            and finite(restorer_receipt['termination_grace_seconds']) == 60
            and restorer_receipt['termination_grace_exhausted'] is False
            and restorer_receipt['termination_signal'] == 'not_attempted'
            and number(restorer_receipt['output_limit_bytes']) == 1024**2
            and number(restorer_receipt['captured_bytes']) == number(restorer_receipt['observed_bytes']) <= 1024**2
            and restorer_receipt['report_capture'] is None and restorer_receipt['report_limit_bytes'] is None
            and restorer_receipt['process_observer_enabled'] is False
            and restorer_receipt['workload_output_qualified'] is False
            and restorer_receipt['executable_identity_verified'] is False, 'outer restoration capture budget/scope differs')
    sha(restorer_receipt['capture_sha256'])
    spawn = finite(restorer_receipt['spawn_returned_monotonic_seconds'])
    require(outer['start'][0]/1e9 <= spawn <= outer['end'][1]/1e9, 'outer spawn outside receipt')
    require(type(ready['schema_version']) is int and ready['schema_version'] == 1
            and ready['kind'] == 'telemetry_ready' and number(ready['worker_pid'],1) == tel['pid']
            and number(ready['gpu_pid'],1) == gpu_pid and ready['observer_affinity'] == [0]
            and all(type(v) is int for v in ready['observer_affinity']), 'readiness worker/GPU/affinity mismatch')
    require(collection['ready_canonical_sha256'] == digest(ready), 'readiness content binding differs')
    recorded, seen = clocks.stamp(ready['record_clock']), clocks.stamp(collection['ready_observed_clock'])
    require(cpu['start'][1] <= recorded[0] <= recorded[1] <= seen[0] <= seen[1] <= outer['start'][0]
            and recorded[1] <= cpu['end'][0], 'readiness does not precede restorer inside CPU lifetime')
    gpu_start, gpu_stop = clocks.stamp(telemetry['gpu_start_clock']), clocks.stamp(telemetry['gpu_stop_clock'])
    require(tel['start'][1] <= gpu_start[0] <= gpu_start[1] <= cpu['start'][0]
            and cpu['end'][1] <= gpu_stop[0] <= gpu_stop[1] <= tel['end'][0],
            'GPU observer launch/cleanup does not enclose CPU capture')
    require(isinstance(cpu_samples,list) and 2 <= len(cpu_samples) <= 901, 'bounded CPU samples required')
    samples = []
    for index,row in enumerate(cpu_samples):
        require(row['kind'] == 'sample' and number(row['index']) == index, 'CPU sample order differs')
        samples.append(clocks.cpu_sample(row))
    require(cpu['start'][1] <= samples[0][0] and samples[-1][1] <= cpu['end'][0]
            and samples[0][1] <= recorded[0] and all(a[1] <= b[0] for a,b in zip(samples,samples[1:])),
            'CPU samples outside receipt/readiness or unordered')
    require(samples[0][1] <= outer['start'][0] and outer['end'][1] <= samples[-1][0],
            'restorer capture not bracketed by CPU samples')
    require(isinstance(sequence['jobs'],list) and len(sequence['jobs']) == 18, 'all eighteen jobs required')
    jobs, prior = [], seq['start'][1]
    for ordinal,(runtime,row) in enumerate(zip(ORDER,sequence['jobs']),1):
        require(number(row['ordinal'],1) == ordinal and row['runtime'] == runtime
                and number(row['repeat'],1) == (ordinal-1)//3+1, 'clock job order differs')
        receipt = row['receipt']; pid = receipt_complete(receipt)
        require(pid not in pids and receipt['process_observer_enabled'] is True, 'duplicate/unobserved workload PID')
        pids.append(pid)
        job = clocks.lifetime(receipt)
        same_process(seq,job); contains(seq,job)
        require(prior <= job['start'][0], 'overlapping job receipts')
        prior = job['end'][1]
        require(samples[0][1] <= job['start'][0] and job['end'][1] <= samples[-1][0],
                'job receipt not bracketed by CPU samples')
        jobs.append({'ordinal':ordinal,'pid':pid,'receipt_canonical_sha256':digest(receipt)})
    offset, other = clocks.offset(), transaction['clock_domain_offset_ns']
    offset['minimum'] = max(offset['minimum'],other['minimum'])
    offset['maximum'] = min(offset['maximum'],other['maximum'])
    require(offset['minimum'] <= offset['maximum'], 'restoration and observer clock domains differ')
    return {'schema_version':1, 'sibling_lifecycle_consistency_verified':True, 'jobs':jobs,
            'restoration_binding':transaction, 'clock_domain_offset_ns':offset,
            'records_canonical_sha256':{name:digest(value) for name,value in (
                ('collection',collection),('restorer_receipt',restorer_receipt),('observer_supervisor',observer_supervisor),
                ('telemetry',telemetry),('ready',ready),('cpu_samples',cpu_samples))},
            'launch_provenance_verified':False, 'telemetry_contents_qualified':False,
            'hardware_policy_restoration_verified':False, 'benchmark_outputs_qualified':False,
            'postflight_call_order_enforced':False,
            'scope':'Supplied sibling parent/PID/lifetime and sample bracketing consistency only. Not live '
                    'ownership, safe caller ancestry, role authenticity, continuous telemetry, hardware restoration, '
                    'process-map/output quality or enforced rehash-last finalization.'}
