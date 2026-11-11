"""Bind recorded process clocks and owned-child lifecycle results, offline only.

No file IO, execution, policy action or postflight authorization. The final entry
must itself reap the supervisor before calling this validator or hashing frozen
inputs. Supplied receipts cannot prove independent producer identity. CPU/GPU
contents, process maps, command/source/raw-file bindings and codec outputs still
require their separate validators. Older receipts without explicit clocks fail.
"""
import hashlib
import json
import math
import signal

from readonly_trace_preflight import require
from readonly_trace_sequence import ORDER
from validate_readonly_telemetry import integer, utc_ns, TIME_QUANTIZATION_NS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Clocks:
    def __init__(self):
        self.offsets = []

    def stamp(self, value):
        require(set(value) == {'monotonic_before_ns', 'monotonic_after_ns', 'utc'}, 'invalid clock fields')
        before = integer(value['monotonic_before_ns'])
        after = integer(value['monotonic_after_ns'])
        require(before <= after, 'reversed clock bracket')
        utc = utc_ns(value['utc'])
        self.offsets.append((utc - after - TIME_QUANTIZATION_NS, utc - before + TIME_QUANTIZATION_NS))
        return before, after

    def lifetime(self, record):
        require(type(record['schema_version']) is int and record['schema_version'] == 1,
                'unsupported lifecycle record schema')
        value = record['lifetime']
        require(set(value) == {'pid', 'parent_pid', 'start', 'end'}, 'invalid lifetime fields')
        pid, parent = integer(value['pid'], 1), integer(value['parent_pid'], 1)
        require(pid != parent, 'self-parent lifetime')
        start, end = self.stamp(value['start']), self.stamp(value['end'])
        require(start[1] <= end[0], 'reversed lifetime')
        if 'started_utc' in record:
            require(utc_ns(value['start']['utc']) <= utc_ns(record['started_utc'])
                    <= utc_ns(record['finished_utc']) <= utc_ns(value['end']['utc']),
                    'receipt UTC outside lifetime')
        if 'elapsed_seconds_including_cleanup' in record:
            elapsed = record['elapsed_seconds_including_cleanup']
            require(type(elapsed) in (int, float) and math.isfinite(elapsed)
                    and 0 <= elapsed <= (end[1] - start[0]) / 1e9,
                    'elapsed time outside lifetime envelope')
        return {'pid': pid, 'parent': parent, 'start': start, 'end': end}

    def cpu_sample(self, row):
        return self.stamp({'monotonic_before_ns': row['monotonic_start_ns'],
                           'monotonic_after_ns': row['monotonic_end_ns'], 'utc': row['utc']})

    def offset(self):
        low, high = max(v[0] for v in self.offsets), min(v[1] for v in self.offsets)
        require(low <= high, 'inconsistent UTC/monotonic clock domain')
        return {'minimum': low, 'maximum': high, 'quantization_ns': TIME_QUANTIZATION_NS}


def contains(parent, child):
    require(parent['start'][1] <= child['start'][0] and child['end'][1] <= parent['end'][0],
            'lifetime is not nested')


def same_process(first, second):
    require((first['pid'], first['parent']) == (second['pid'], second['parent']), 'collector identity mismatch')


def receipt_complete(receipt):
    require(receipt['status'] == 'complete' and receipt['issues'] == []
            and type(receipt['returncode']) is int and receipt['returncode'] == 0
            and receipt['direct_child_reaped'] is True and receipt['capture_eof'] is True
            and receipt['group_kill'] in ('sent', 'no_group') and receipt['interrupt_signal'] is None,
            'incomplete owned-child receipt')
    return integer(receipt['pid'], 1)


def validate(supervisor_receipt, supervisor, collection, sequence, telemetry, ready, observations, cpu_samples):
    """Check clocks/parent chains for a fully successful fixed twelve-job collection.

    cpu_samples are the actual parsed sample rows, not derived endpoint estimates;
    full 901-sample arithmetic/source/duration qualification is deliberately separate.
    Lifetime stamps bracket setup through result construction, NOT exact exec/exit.
    """
    require(type(ready['schema_version']) is int and ready['schema_version'] == 1
            and ready['kind'] == 'telemetry_ready', 'unsupported readiness record')
    clocks = Clocks()
    outer = clocks.lifetime(supervisor_receipt)
    sup, worker = clocks.lifetime(supervisor), clocks.lifetime(supervisor['worker'])
    coord, seq = clocks.lifetime(collection), clocks.lifetime(sequence)
    observer, cpu = clocks.lifetime(telemetry), clocks.lifetime(telemetry['cpu_receipt'])
    require(receipt_complete(supervisor_receipt) == sup['pid'] and sup['parent'] == outer['pid'],
            'supervisor parent/receipt mismatch')
    require(receipt_complete(supervisor['worker']) == coord['pid'] and coord['parent'] == sup['pid'],
            'coordinator parent/receipt mismatch')
    same_process(worker, sup)
    same_process(seq, coord)
    same_process(cpu, observer)
    require(observer['parent'] == coord['pid'] and collection['observer_pid'] == observer['pid']
            and type(collection['observer_pid']) is int
            and ready['worker_pid'] == observer['pid'] and type(ready['worker_pid']) is int,
            'observer parent/readiness mismatch')
    cpu_pid, gpu_pid = receipt_complete(telemetry['cpu_receipt']), integer(ready['gpu_pid'], 1)
    require(type(telemetry['gpu_pid']) is int and telemetry['gpu_pid'] == gpu_pid,
            'GPU child readiness/result PID mismatch')
    gpu_start, gpu_stop = clocks.stamp(telemetry['gpu_start_clock']), clocks.stamp(telemetry['gpu_stop_clock'])
    require(observer['start'][1] <= gpu_start[0] and gpu_start[1] <= cpu['start'][0]
            and cpu['end'][1] <= gpu_stop[0] and gpu_stop[1] <= observer['end'][0],
            'GPU child launch/cleanup does not enclose CPU capture')
    pids = [outer['pid'], sup['pid'], coord['pid'], observer['pid'], cpu_pid, gpu_pid]
    require(len(set(pids)) == len(pids), 'duplicate collector/observer PIDs')
    contains(outer, sup)
    contains(sup, worker)
    contains(worker, coord)
    contains(coord, observer)
    contains(coord, seq)
    contains(observer, cpu)
    require(supervisor['status'] == 'complete' and supervisor['errors'] == []
            and supervisor['parent_lost'] is False and supervisor['orphan_observed'] is False
            and supervisor['subreaper_restored'] is True and supervisor['interrupt_signal'] is None,
            'supervisor did not complete cleanly')
    cleanup = supervisor['cleanup']
    require(cleanup['complete'] is True and all(cleanup[name] == [] for name in
            ('errors', 'remaining', 'signalled', 'reaped')), 'supervisor orphan cleanup is not success')
    require(collection['status'] == 'complete' and collection['errors'] == []
            and collection['interrupt_signal'] is None and collection['sequence_status'] == 'complete'
            and collection['telemetry_status'] == 'complete', 'incomplete coordinator')
    shutdown = collection['observer_shutdown']
    require(shutdown['verified'] is True and type(shutdown['returncode']) is int
            and shutdown['returncode'] == 0 and shutdown['kill_escalation'] is False,
            'observer not cleanly reaped')
    require(telemetry['status'] == 'complete' and telemetry['ready'] is True
            and telemetry['errors'] == [] and telemetry['interrupt_signal'] is None, 'incomplete telemetry worker')
    shutdown = telemetry['gpu_shutdown']
    require(shutdown['verified'] is True and type(shutdown['returncode']) is int
            and shutdown['returncode'] in (0, -signal.SIGTERM) and shutdown['kill_escalation'] is False,
            'GPU observer not cleanly stopped')
    require(sequence['status'] == 'complete' and sequence['stop_reason'] is None
            and len(sequence['jobs']) == len(observations) == len(ORDER), 'incomplete fixed sequence')
    require(collection['ready_canonical_sha256'] == digest(ready), 'readiness content binding mismatch')
    recorded, observed = clocks.stamp(ready['record_clock']), clocks.stamp(collection['ready_observed_clock'])
    require(cpu['start'][1] <= recorded[0] and recorded[1] <= observed[0]
            and observed[1] <= seq['start'][0] and recorded[1] <= cpu['end'][0],
            'readiness outside observer lifetime or after sequence start')
    require(isinstance(cpu_samples, list) and len(cpu_samples) >= 2, 'CPU endpoints unavailable')
    samples = [clocks.cpu_sample(row) for row in cpu_samples]
    require(cpu['start'][1] <= samples[0][0] and samples[-1][1] <= cpu['end'][0]
            and samples[0][1] <= recorded[0], 'CPU samples not inside sampler receipt/readiness')
    require(all(a[1] <= b[0] for a, b in zip(samples, samples[1:])), 'unordered CPU samples')
    prior_end = seq['start'][1]
    bindings = []
    for ordinal, (condition, job, observation) in enumerate(zip(ORDER, sequence['jobs'], observations), 1):
        require(type(job['ordinal']) is int and job['ordinal'] == ordinal and job['condition'] == condition,
                'job order mismatch')
        receipt = job['receipt']
        pid = receipt_complete(receipt)
        require(type(observation['schema_version']) is int and observation['schema_version'] == 1
                and pid not in pids and type(observation['pid']) is int and observation['pid'] == pid
                and observation['status'] == 'complete' and observation['issues'] == []
                and receipt['process_observer_enabled'] is True, 'missing/mismatched process observation')
        pids.append(pid)
        life = clocks.lifetime(receipt)
        same_process(life, coord)
        contains(seq, life)
        require(prior_end <= life['start'][0], 'overlapping/out-of-order job receipts')
        prior_end = life['end'][1]
        require(samples[0][1] <= life['start'][0] and life['end'][1] <= samples[-1][0],
                'job not bracketed by CPU sample reads')
        require(isinstance(observation['samples'], list) and observation['samples'], 'no process samples')
        previous = life['start'][1]
        for sample in observation['samples']:
            start, end = integer(sample['monotonic_start_ns']), integer(sample['monotonic_end_ns'])
            require(previous <= start <= end <= life['end'][0], 'process sample outside receipt/order')
            previous = end
        bindings.append({'ordinal': ordinal, 'pid': pid, 'receipt_canonical_sha256': digest(receipt),
                         'observation_canonical_sha256': digest(observation)})
    return {'schema_version': 1, 'lifecycle_consistency_verified': True, 'jobs': bindings,
            'utc_minus_monotonic_ns': clocks.offset(),
            'records_canonical_sha256': {name: digest(value) for name, value in (
                ('supervisor_receipt', supervisor_receipt), ('supervisor', supervisor), ('collection', collection),
                ('sequence', sequence), ('telemetry', telemetry), ('ready', ready), ('cpu_samples', cpu_samples))},
            'launch_provenance_verified': False, 'telemetry_contents_qualified': False,
            'benchmark_outputs_qualified': False, 'postflight_call_order_enforced': False,
            'scope': 'Supplied clocks/parent identities/lifecycle consistency only. Does not prove '
                     'independent production, exact exec/exit, immutable bytes, full sampling contents, '
                     'live completion at the caller, or absence of supervisor-SIGKILL/host-failure hazards.'}
