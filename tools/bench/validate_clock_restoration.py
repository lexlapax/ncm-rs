"""Offline restoration-record consistency, not hardware restoration or launch safety.

No file IO or command execution. The eventual finalizer must bind these records to
owned outer receipts and sibling observer lifetimes before final frozen rehash.
"""
import math
from pathlib import Path

from readonly_trace_preflight import require
from validate_readonly_lifecycle import Clocks, contains, same_process, receipt_complete, digest
from verify_readonly_processes import number
from verify_clock_processes import sha


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite deadline/time required')
    return value


def validate(restoration, obligation, sequence, *, commands, cwd, work_deadline,
             external_cancel_check_enabled):
    """Check a successful production-budget transaction against separate declarations.

    Failed attempts/retries remain failed even if a reset eventually succeeds.
    Commands/cwd and the original deadline must be bound by the future reviewed
    entry/handoff, not reconstructed from these records to make them agree.
    No set/reset return code establishes enforcement or restored hardware policy.
    """
    require(type(external_cancel_check_enabled) is bool, 'explicit cancellation hook contract required')
    require(set(commands) == {'set','worker','reset'} and isinstance(cwd, str)
            and Path(cwd).is_absolute() and '..' not in Path(cwd).parts, 'explicit commands/cwd required')
    for argv in commands.values():
        require(isinstance(argv, list) and argv and Path(argv[0]).is_absolute()
                and all(isinstance(a, str) and a and '\0' not in a for a in argv), 'invalid declared command')
    for record in (restoration, obligation):
        require(type(record['schema_version']) is int and record['schema_version'] == 2,
                'restoration schema 2 required')
    require(restoration['status'] == 'complete' and restoration['armed'] is True
            and restoration['errors'] == [] and restoration['parent_lost'] is False
            and restoration['orphan_observed'] is False and restoration['external_cancelled'] is False
            and restoration['external_cancel_check_enabled'] is external_cancel_check_enabled
            and restoration['interrupt_signal'] is None and restoration['subreaper_restored'] is True
            and restoration['reset_command_succeeded'] is True
            and restoration['reset_request_within_deadline'] is True, 'incomplete restoration transaction')
    require(all(restoration[k] is False for k in ('hardware_policy_restoration_verified',
            'clock_enforcement_verified','launch_provenance_verified','benchmark_outputs_qualified')),
            'unsupported stronger restoration claim')
    require(obligation['armed_before_set'] is True
            and type(obligation['maximum_reset_attempts']) is int and obligation['maximum_reset_attempts'] == 2
            and finite(obligation['lock_seconds']) == 600 and finite(obligation['command_seconds']) == 10
            and finite(obligation['cleanup_reserve_seconds']) == finite(restoration['cleanup_reserve_seconds']) == 10,
            'fixed restoration budgets differ')
    work = finite(work_deadline)
    reset = finite(obligation['reset_deadline_monotonic_seconds'])
    require(work > 0 and finite(obligation['work_deadline_monotonic_seconds'])
            == finite(restoration['work_deadline_monotonic_seconds']) == work == reset - 10
            and finite(restoration['reset_deadline_monotonic_seconds']) == reset,
            'original work/reset deadline mismatch')
    require(all(obligation[k+'_argv'] == commands[k] for k in commands), 'obligation command mismatch')
    clocks = Clocks()
    # Only the outer record version differs; use the shared strict lifetime parser
    # without accepting schema 2 for unrelated receipt/sequence records.
    life = clocks.lifetime({'schema_version': 1, 'lifetime': restoration['lifetime']})
    require(number(obligation['supervisor_pid'], 1) == life['pid']
            and number(obligation['supervisor_parent_pid'], 1) == life['parent'], 'obligation writer mismatch')
    armed = clocks.stamp(obligation['clock'])
    require(life['start'][1]/1e9 <= reset - 600 <= armed[0]/1e9
            and armed[1] <= life['end'][0], 'armed clock/deadline outside supervisor lifetime')
    resets = restoration['reset_attempts']
    require(isinstance(resets, list) and len(resets) == 1, 'successful transaction cannot conceal reset retry')
    require(isinstance(restoration['cleanups'], list)
            and [r['after'] for r in restoration['cleanups']] == ['set','before_reset','reset-1'],
            'cleanup order/coverage mismatch')
    for row in restoration['cleanups']:
        cleanup = row['result']
        require(cleanup['complete'] is True and all(cleanup[k] == []
                for k in ('signalled','reaped','remaining','errors')), 'owned orphan cleanup is not success')
    pids = {life['pid'], life['parent']}
    envelopes, receipts = {}, {'set':restoration['set'], 'worker':restoration['worker'], 'reset':resets[0]}
    for label, receipt in receipts.items():
        pid = receipt_complete(receipt)
        require(pid not in pids, 'duplicate transaction PID')
        pids.add(pid)
        child = clocks.lifetime(receipt)
        same_process(life, child); contains(life, child)
        envelopes[label] = child
        require(receipt['argv'] == commands[label] and receipt['cwd'] == cwd
                and finite(receipt['timeout_seconds']) == (600 if label == 'worker' else 10)
                and number(receipt['output_limit_bytes']) == 1024**2
                and finite(receipt['termination_grace_seconds']) == 0
                and receipt['termination_grace_exhausted'] is False
                and receipt['termination_signal'] == 'not_attempted'
                and receipt['process_observer_enabled'] is False
                and receipt['report_capture'] is None and receipt['report_limit_bytes'] is None
                and receipt['workload_output_qualified'] is False and receipt['executable_identity_verified'] is False,
                'transaction receipt command/limits/scope mismatch')
        require(number(receipt['captured_bytes']) == number(receipt['observed_bytes']) <= 1024**2,
                'transaction capture accounting mismatch')
        sha(receipt['capture_sha256'])
        spawn = finite(receipt['spawn_returned_monotonic_seconds'])
        require(child['start'][0]/1e9 <= spawn <= child['end'][1]/1e9,
                'spawn timestamp outside receipt')
        if label != 'reset':
            require(child['end'][1]/1e9 <= work, 'work finished after original deadline')
    reset_start = clocks.stamp(restoration['reset_started_clock'])
    set_life, worker, reset_life = (envelopes[k] for k in ('set','worker','reset'))
    require(armed[1] <= set_life['start'][0] and set_life['end'][1] <= worker['start'][0]
            and worker['end'][1] <= reset_start[0] <= reset_start[1] <= reset_life['start'][0],
            'set/worker/reset lifetime ordering differs')
    require(reset-600 <= receipts['reset']['spawn_returned_monotonic_seconds'] <= reset,
            'first reset request missed original deadline')
    require(type(sequence['schema_version']) is int and sequence['schema_version'] == 1
            and sequence['status'] == 'complete' and sequence['stop_reason'] is None
            and sequence['external_cancelled'] is False
            and finite(sequence['work_deadline_monotonic_seconds']) == work, 'sequence original deadline/status differs')
    seq = clocks.lifetime(sequence)
    require(seq['pid'] == receipts['worker']['pid'] and seq['parent'] == life['pid'],
            'sequence worker parent/PID mismatch')
    contains(worker, seq)
    require(seq['end'][1]/1e9 <= work, 'sequence exceeded original work deadline')
    return {'schema_version': 1, 'restoration_record_consistency_verified': True,
            'supervisor_pid': life['pid'], 'worker_pid': seq['pid'],
            'set_pid': receipts['set']['pid'], 'reset_pid': receipts['reset']['pid'],
            'work_deadline_monotonic_seconds': work, 'reset_deadline_monotonic_seconds': reset,
            'clock_domain_offset_ns': clocks.offset(),
            'records_canonical_sha256': {name:digest(value) for name,value in (
                ('restoration',restoration),('obligation',obligation),('sequence',sequence))},
            'hardware_policy_restoration_verified': False, 'clock_enforcement_verified': False,
            'outer_supervision_verified': False, 'observer_coverage_verified': False,
            'launch_provenance_verified': False, 'benchmark_outputs_qualified': False,
            'postflight_call_order_enforced': False,
            'scope': 'Supplied successful obligation/transaction/worker clocks and declarations only. '
                     'Not command authenticity, exact exec/driver timing, hardware policy restoration, '
                     'observer coverage, outer reaping, complete job validation or finalization ordering.'}
