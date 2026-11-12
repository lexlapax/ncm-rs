"""Bind current control-side files to separate role declarations, offline only.

No launch, process inspection, policy action or frozen-file rehash. Lifecycle/content
validators still owe actual semantics; a future entry must authenticate role plans.
"""
from copy import deepcopy
from pathlib import Path

from clock_evidence import META_BYTES
from readonly_trace_preflight import require
from validate_clock_restoration import finite
from validate_readonly_lifecycle import digest, receipt_complete
from verify_readonly_processes import number, key

ROLES = ('restorer','set','worker','reset','telemetry_worker','cpu')
FIELDS = ('argv','cwd','timeout_seconds','termination_grace_seconds','output_limit_bytes','report_limit_bytes')
CAPTURE_BYTES = 1024**2


def read_control(evidence,*,roles):
    """Read 21 control files; supplied roles must come from reviewed launch planning.

    Exact receipt-vs-declaration comparison is not producer authentication. This
    reader does not choose observer timeouts or prove safe coordinator ancestry.
    Observer-supervisor stdout has no producer hash receipt and stays post-only.
    """
    roles=deepcopy(roles)
    require(isinstance(roles,dict) and set(roles)==set(ROLES), 'complete control role declarations required')
    for role in roles.values():
        require(set(role)==set(FIELDS) and isinstance(role['argv'],list) and role['argv']
                and all(isinstance(v,str) and v and '\0' not in v for v in role['argv'])
                and Path(role['argv'][0]).is_absolute() and isinstance(role['cwd'],str)
                and Path(role['cwd']).is_absolute() and '..' not in Path(role['cwd']).parts
                and '\0' not in role['cwd'], 'explicit role commands/cwd required')
        require(finite(role['timeout_seconds'])>0 and 0<=finite(role['termination_grace_seconds'])<=60
                and number(role['output_limit_bytes'])==CAPTURE_BYTES
                and (role['report_limit_bytes'] is None or number(role['report_limit_bytes'],1)<=META_BYTES),
                'invalid declared role capture bounds')
    require(roles['cpu']['report_limit_bytes']==META_BYTES
            and all(roles[r]['report_limit_bytes'] is None for r in ROLES if r!='cpu'),
            'only CPU role has a telemetry report')
    collection=evidence.record('collection.json')
    restorer=evidence.record('restorer.receipt.json',CAPTURE_BYTES)
    restoration=evidence.record('restoration/restoration.json')
    obligation=evidence.record('restoration/obligation.json',CAPTURE_BYTES)
    supervisor=evidence.record('observer-supervision/supervisor.json')
    telemetry=evidence.record('telemetry/telemetry.json')
    ready=evidence.record('telemetry/ready.json')
    for record in (collection,supervisor,telemetry):
        require(type(record['schema_version']) is int and record['schema_version']==1
                and record['status']=='complete', 'incomplete control record')
    require(type(restoration['schema_version']) is int and restoration['schema_version']==2
            and restoration['status']=='complete' and isinstance(restoration['reset_attempts'],list)
            and len(restoration['reset_attempts'])==1, 'incomplete/retried restoration files')
    require(type(obligation['schema_version']) is int and obligation['schema_version']==2
            and obligation['armed_before_set'] is True
            and all(obligation[k+'_argv']==roles[k]['argv'] for k in ('set','worker','reset')),
            'obligation command declarations differ')
    items=(('restorer','restorer',restorer),('set','restoration/set',restoration['set']),
           ('worker','restoration/worker',restoration['worker']),
           ('reset','restoration/reset-1',restoration['reset_attempts'][0]),
           ('telemetry_worker','observer-supervision/worker',supervisor['worker']),
           ('cpu','telemetry/cpu-command',telemetry['cpu_receipt']))
    receipts,bindings,pids={},[],set()
    for name,prefix,embedded in items:
        receipt=restorer if name=='restorer' else evidence.record(prefix+'.receipt.json',CAPTURE_BYTES)
        require(digest(receipt)==digest(embedded), 'individual/embedded control receipt differs')
        pid=receipt_complete(receipt)
        require(type(receipt['schema_version']) is int and receipt['schema_version']==1 and pid not in pids,
                'duplicate/invalid control child receipt')
        pids.add(pid)
        require(digest({k:receipt[k] for k in FIELDS})==digest(roles[name]), 'control receipt differs from declared role')
        require(receipt['termination_signal']=='not_attempted' and receipt['termination_grace_exhausted'] is False
                and receipt['process_observer_enabled'] is False and receipt['workload_output_qualified'] is False
                and receipt['executable_identity_verified'] is False, 'unsupported control receipt scope/termination')
        path=prefix+('.capture' if name=='restorer' else '.txt')
        raw=evidence.raw(path,CAPTURE_BYTES); identity=evidence.inputs[path]
        require(number(receipt['captured_bytes'])==number(receipt['observed_bytes'])==len(raw)
                and receipt['capture_sha256']==identity['sha256'], 'control capture differs from receipt')
        if name!='cpu': require(receipt['report_capture'] is None,'unexpected control report')
        receipts[name]=receipt
        bindings.append({'role':name,'pid':pid,'receipt_canonical_sha256':digest(receipt),'capture':identity.copy(),
                         'producer_capture_inode_verified':False})
    cpu=evidence.raw('telemetry/cpu.jsonl'); cpu_identity=evidence.inputs['telemetry/cpu.jsonl']
    report=receipts['cpu']['report_capture']
    require(report['path']==str(evidence.root/'telemetry/cpu.jsonl') and key(report)==key(cpu_identity)
            and number(report['size_bytes'],1)==len(cpu) and report['sha256']==cpu_identity['sha256']
            and report['post_exit_hash_verified'] is True and report['producer_identity_verified'] is False,
            'CPU telemetry report file differs from receipt')
    gpu=evidence.raw('telemetry/gpu.csv')
    require(telemetry['gpu_sha256']==evidence.inputs['telemetry/gpu.csv']['sha256'], 'GPU telemetry file digest differs')
    evidence.raw('observer-worker.txt',CAPTURE_BYTES)
    evidence.verify_unchanged()
    return {'collection':collection,'restorer_receipt':restorer,'restoration':restoration,'obligation':obligation,
            'observer_supervisor':supervisor,'telemetry':telemetry,'ready':ready,'cpu_raw':cpu,'gpu_raw':gpu,
            'binding':{'schema_version':1,'control_file_receipt_consistency_verified':True,'roles':bindings,
                       'declared_roles_canonical_sha256':digest(roles),
                       'observer_supervisor_capture':{'identity':evidence.inputs['observer-worker.txt'].copy(),
                                                      'binding':'post_only','producer_digest_verified':False},
                       'cpu_report':cpu_identity.copy(),'gpu_capture':evidence.inputs['telemetry/gpu.csv'].copy(),
                       'launch_provenance_verified':False,'lifecycle_qualified':False,'telemetry_contents_qualified':False,
                       'hardware_policy_restoration_verified':False,'postflight_call_order_enforced':False,
                       'scope':'Current private control files and supplied role-receipt declarations only. '
                               'CPU report pathname identity and capture digests checked; other producer capture '
                               'inodes and observer-supervisor stdout digest unavailable. Not live provenance, '
                               'safe ancestry, full contents/lifecycle, policy restoration or finalization order.'}}
