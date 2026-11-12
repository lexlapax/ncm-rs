"""Same-evidence clock postflight after ONE trusted restoration-aware runner.

Import-only integration, not launch authorization or a sandbox. The future frozen
entry owes full preflight, actual exclusive reaping of both sibling branches,
policy/permission/availability checks and authenticated role declarations. Never
wrap the coordinator/restorer in generic adoption-and-SIGKILL supervision here.
"""
import hashlib
import json
import os
from pathlib import Path

from clock_evidence import Evidence, META_BYTES, private, read_sequence
from clock_control_evidence import read_control
from readonly_trace_preflight import require
from validate_readonly_telemetry import unique_object, reject_constant
from validate_readonly_lifecycle import digest
from validate_clock_restoration import finite
from validate_clock_lifecycle import validate as validate_lifecycle
from validate_clock_telemetry import validate as validate_telemetry
from verify_clock_processes import validate_processes
from verify_clock_files import canonical_spelling, declared_inventory, verify_files
from qualify_clock_codec import qualify

FIELDS = {'schema_version','kind','collection_root','jobs','roles','library_contract','injection',
          'frozen_identities','path_bindings','library_roots','sources','python_metadata','cpu_source',
          'gpu_uuid','timezone_name','cwd','external_cancel_check_enabled'}


def parsed(raw):
    return json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)


def publish(directory,name,raw):
    """Exclusive private partial output; completion marker is linked only after close."""
    fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        require(stream.write(raw)==len(raw),'incomplete finalizer output write')
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def encoded(value):
    return json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'


def run_then_finalize(inputs_raw,expected_sha256,launch_dir,run_collection):
    """Freeze explicit inputs, invoke runner once, gate the same bytes, rehash LAST.

    Runner creates the fresh collection directory and returns {'collection': its
    coordinator result, 'work_deadline': the original armed deadline}, only after
    both branches are stopped/reaped. That is a trusted-code contract, not something
    a return dictionary can prove. This helper does not prepare or launch roles.
    A failed attempt leaves its private launch directory occupied; no retry path.
    """
    require(isinstance(inputs_raw,bytes) and 0<len(inputs_raw)<=META_BYTES
            and hashlib.sha256(inputs_raw).hexdigest()==expected_sha256,'finalizer input digest/size differs')
    inputs=parsed(inputs_raw)
    require(isinstance(inputs,dict) and set(inputs)==FIELDS and type(inputs['schema_version']) is int
            and inputs['schema_version']==1 and inputs['kind']=='clock-finalization-inputs'
            and type(inputs['external_cancel_check_enabled']) is bool and callable(run_collection),
            'explicit finalization inputs and trusted runner required')
    # Pure declaration admission, not early frozen-file hashing. Never spend a
    # collection on an inventory already known to exceed unchanged postflight caps.
    declared_inventory(inputs['library_contract'],frozen_identities=inputs['frozen_identities'],
                       path_bindings=inputs['path_bindings'],injection=inputs['injection'])
    collection=canonical_spelling(inputs['collection_root']); launch=canonical_spelling(str(launch_dir))
    require(collection.parent.resolve(strict=True)==collection.parent and launch.parent.resolve(strict=True)==launch.parent
            and not os.path.lexists(collection) and not os.path.lexists(launch)
            and not collection.is_relative_to(launch) and not launch.is_relative_to(collection),
            'fresh disjoint canonical collection/launch paths required')
    # Not full launch preflight: that remains the reviewed entry's responsibility.
    launch.mkdir(mode=0o700); private(launch.stat(),True)
    publish(launch,'finalizer-inputs.json',inputs_raw)
    returned=run_collection()  # Sole submission, never retried or generically supervised.
    require(isinstance(returned,dict) and set(returned)=={'collection','work_deadline'},'invalid runner return')
    deadline=finite(returned['work_deadline']); require(deadline>0,'original armed deadline required')
    with Evidence(launch) as launch_evidence, Evidence(collection) as evidence:
        require(launch_evidence.raw('finalizer-inputs.json')==inputs_raw,'finalization inputs changed during runner')
        seq=read_sequence(evidence,jobs=inputs['jobs'],work_deadline=deadline)
        control=read_control(evidence,roles=inputs['roles'])
        require(digest(control['collection'])==digest(returned['collection']),'returned/persisted coordinator differs')
        sequence=seq['sequence']
        contents=validate_telemetry(control['cpu_raw'],control['gpu_raw'],sequence,control['restorer_receipt'],
            control['telemetry'],control['ready'],expected_cpu_source=inputs['cpu_source'],gpu_uuid=inputs['gpu_uuid'],
            timezone_name=inputs['timezone_name'],cpu_path=str(collection/'telemetry/cpu.jsonl'),
            cpu_argv=inputs['roles']['cpu']['argv'])
        require(all(contents[k] is True for k in ('telemetry_contents_validated','readiness_contents_bound',
                'receipt_windows_bracketed')),'telemetry validation incomplete')
        samples=[parsed(line) for line in control['cpu_raw'].splitlines()][1:-1]
        lifecycle=validate_lifecycle(control['collection'],control['restorer_receipt'],control['restoration'],
            control['obligation'],sequence,control['observer_supervisor'],control['telemetry'],control['ready'],samples,
            commands={k:inputs['roles'][k]['argv'] for k in ('set','worker','reset')},cwd=inputs['cwd'],
            work_deadline=deadline,external_cancel_check_enabled=inputs['external_cancel_check_enabled'])
        require(lifecycle['sibling_lifecycle_consistency_verified'] is True
                and lifecycle['restoration_binding']['restoration_record_consistency_verified'] is True,
                'sibling/restoration validation incomplete')
        processes=validate_processes(inputs['jobs'],sequence,inputs['library_contract'],
            injection=inputs['injection'],work_deadline=deadline)
        require(processes['process_receipt_consistency_verified'] is True,'process validation incomplete')
        outputs=qualify(seq['raw_reports'],sequence['jobs'],sources=inputs['sources'],python_metadata=inputs['python_metadata'])
        require(outputs['reported_codec_outputs_qualified'] is True and outputs['performance_acceptance_evaluated'] is False,
                'report qualification incomplete or overclaimed')
        # Every validator must bind the same ordered receipts; no transplanted gate.
        expected=[digest(row['receipt']) for row in sequence['jobs']]
        for gate in (contents,lifecycle,processes,outputs):
            require([r['receipt_canonical_sha256'] for r in gate['jobs']]==expected,'validator receipt bindings differ')
        offsets=[gate['clock_domain_offset_ns'] for gate in (contents,lifecycle,processes)]
        offset={'minimum':max(v['minimum'] for v in offsets),'maximum':min(v['maximum'] for v in offsets)}
        require(offset['minimum']<=offset['maximum'],'composed clock domains disagree')
        evidence.verify_unchanged(); launch_evidence.verify_unchanged()
        # LAST input validation/file access stage. All collectors must already be
        # reaped by the trusted runner. Only derived-output publication follows.
        postflight=verify_files(processes,inputs['library_contract'],frozen_identities=inputs['frozen_identities'],
            path_bindings=inputs['path_bindings'],injection=inputs['injection'],
            library_roots=tuple(map(Path,inputs['library_roots'])))
        require(all(postflight[k] is True for k in ('supplied_frozen_file_bytes_rechecked',
                'current_mapped_paths_verified','supplied_path_bindings_rechecked'))
                and postflight['mapped_file_bytes_verified'] is False and postflight['launch_provenance_verified'] is False,
                'frozen postflight incomplete or overclaimed')
        consumed=dict(evidence.inputs); launch_inputs=dict(launch_evidence.inputs)
    output=launch/'completion'; output.mkdir(mode=0o700)
    results={name:publish(output,name,encoded(value)) for name,value in (
        ('evidence-binding.json',{'sequence':seq['binding'],'control':control['binding']}),
        ('telemetry-validation.json',contents),('lifecycle-validation.json',lifecycle),
        ('process-validation.json',processes),('output-qualification.json',outputs),('postflight.json',postflight))}
    result={'schema_version':1,'status':'complete','diagnostic_only':True,'finalizer_inputs_sha256':expected_sha256,
        'runner_return_canonical_sha256':digest(returned),'work_deadline_monotonic_seconds':deadline,
        'consumed_raw_files':consumed,'launch_inputs':launch_inputs,'results':results,'clock_domain_offset_ns':offset,
        'validation_composition_complete':True,'frozen_rehash_after_runner_and_validation':True,
        'launch_provenance_verified':False,'runner_reaping_independently_verified':False,
        'hardware_policy_restoration_verified':False,'clock_enforcement_verified':False,
        'complete_preparation_verified':False,'operator_trace_parity_verified':False,
        'performance_acceptance_evaluated':False,
        'scope':'One trusted runner return, same bounded evidence through all required validators, frozen file '
                'rehash last. Supplied declarations are not complete authenticated preparation; runner/entry '
                'still owe live ownership, safe ancestry, permissions and availability. Observer-supervisor '
                'stdout is post-only. Not immutable mapped bytes, clock enforcement, hardware restoration '
                'or Phase 2 performance acceptance.'}
    # A failed/short write or failed close must never expose completion.json. Keep
    # staging bytes as partial evidence; link is exclusive and never overwrites.
    publish(output,'completion.pending.json',encoded(result))
    os.link(output/'completion.pending.json',output/'completion.json',follow_symlinks=False)
    return result
