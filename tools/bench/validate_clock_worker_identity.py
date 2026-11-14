"""Offline identity/receipt consistency; never authenticates a producer."""
from clock_unprivileged_identity import require
from clock_worker_identity_evidence import target
from validate_readonly_lifecycle import Clocks, contains, digest, receipt_complete


def validate(record, expected, worker_receipt, sequence):
    """Compare supplied identity evidence with a declared target and worker clocks.

    This checks record consistency only: it cannot authenticate the expected
    target, evidence producer, reaping, continuous identity or GPU policy.
    """
    require(isinstance(expected, dict) and set(expected) == {'uid','gid','groups','require_no_new_privs'},
            'explicit identity declaration required')
    declared = target(expected['uid'], expected['gid'], expected['groups'], expected['require_no_new_privs'])
    fields = {'schema_version','kind','target','no_new_privs','lifetime',
              'sampled_unprivileged_identity_verified','identity_declaration_authenticated',
              'continuous_identity_verified','launch_ready'}
    require(isinstance(record, dict) and set(record) == fields
            and type(record['schema_version']) is int and record['schema_version'] == 1
            and record['kind'] == 'clock-worker-identity' and record['target'] == declared
            and type(record['no_new_privs']) is bool
            and (not declared['require_no_new_privs'] or record['no_new_privs'])
            and record['sampled_unprivileged_identity_verified'] is True
            and record['identity_declaration_authenticated'] is False
            and record['continuous_identity_verified'] is False and record['launch_ready'] is False,
            'worker identity record or scope differs')
    require(isinstance(record['target'], dict) and set(record['target']) == set(declared)
            and target(**record['target']) == declared, 'invalid recorded target types')
    clocks = Clocks()
    owner = clocks.lifetime(worker_receipt)
    sample = clocks.lifetime(record)
    seq = clocks.lifetime(sequence)
    require(receipt_complete(worker_receipt) == sample['pid'] == seq['pid']
            and sample['parent'] == seq['parent'] == owner['pid'], 'worker identity PID/parent differs')
    contains(owner, sample)
    contains(owner, seq)
    require(sample['end'][1] <= seq['start'][0], 'worker identity sample does not precede sequence')
    return {'schema_version':1, 'worker_identity_record_consistent':True,
            'record_canonical_sha256':digest(record), 'clock_domain_offset_ns':clocks.offset(),
            'identity_declaration_authenticated':False, 'producer_authenticated':False,
            'continuous_identity_verified':False, 'launch_ready':False}
