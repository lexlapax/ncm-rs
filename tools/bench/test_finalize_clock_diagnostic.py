"""All real gates in fresh children: archived outputs, synthetic clocks/maps/policy.

No GPU collection, execution of ELF stubs or actual ownership/provenance proof.
"""
from copy import deepcopy
from datetime import datetime
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import finalize_clock_diagnostic as finalizer
import test_validate_clock_lifecycle as life
import test_verify_clock_processes as processes
import test_validate_readonly_telemetry as telemetry
from test_qualify_clock_codec import archived_inputs
from test_clock_evidence import write, record
from clock_control_evidence import FIELDS
from clock_codec_sequence import CAPTURE_BYTES, REPORT_BYTES, MIN_FREE_BYTES
from readonly_trace_preflight import file_identity
from validate_readonly_lifecycle import digest


def data(root,archive):
    """Prepare declarations without creating collection; return one fake writer."""
    directory=root/'collection'; frozen=root/'frozen'; frozen.mkdir(mode=0o700)
    pf=processes.ClockProcessValidationTests(); pf.setUp()
    lf=life.ClockLifecycleTests(); lf.setUp()
    identities={}
    for name in ('previous','python','candidate','taskset','required.so','optional.so','injection.so','payload'):
        path=frozen/name; write(path,b'\x7fELF'+name.encode())
        identities[name]=file_identity(path)
    contract=pf.contract
    contract.update(taskset=identities['taskset'],known_libraries=[identities['required.so'],identities['optional.so']],
        libraries={r:[identities['required.so']] for r in contract['libraries']},path_bindings=[],
        workloads={r:dict(command_path=identities[r]['path'],image=identities[r]) for r in contract['workloads']})
    raw_reports,_,sources,metadata=archive
    sequence=pf.result; sequence.update(lifetime=lf.sequence['lifetime'],work_deadline_monotonic_seconds=691)
    jobs=pf.jobs
    for index,(job,row,temporal) in enumerate(zip(jobs,sequence['jobs'],lf.sequence['jobs'])):
        runtime=job['runtime']; kind='python' if runtime=='python' else 'rust'
        path=directory/f'sequence/{runtime}/codec-{kind}-{job["repeat"]}.json'
        job.update(report=str(path),cwd=str(root),argv=[identities['taskset']['path'],'-c','2',identities[runtime]['path'],str(path)])
        r=row['receipt']; timed=temporal['receipt']
        for k in ('pid','lifetime','started_utc','finished_utc','elapsed_seconds_including_cleanup','spawn_returned_monotonic_seconds'):
            r[k]=deepcopy(timed[k])
        r.update(argv=job['argv'],cwd=job['cwd'],capture_sha256=hashlib.sha256(b'').hexdigest())
        o=row['observation']; o['pid']=r['pid']
        files=[]
        for identity in (identities[runtime],identities['required.so']):
            files.append(dict(path=identity['path'],device=28,inode=identity['inode'],path_device=identity['device'],
                              path_inode=identity['inode'],kernel_map_link_verified=True))
        mapping=dict(files=sorted(files,key=lambda v:v['path']),non_file_executable_mappings=['[vdso]'])
        o['mapping_sets']={digest(mapping):mapping}; sample=o['samples'][0]
        start=r['lifetime']['start']['monotonic_before_ns']
        sample.update(mapping_sha256=digest(mapping),monotonic_start_ns=start+20_000_000,monotonic_end_ns=start+21_000_000)
    lf.collection['lifetime']['end']=processes.stamp(1010*10**9)
    lf.supervisor['lifetime']['end']=processes.stamp(1004*10**9)
    def end_receipt(r,end):
        r['lifetime']['end']=processes.stamp(end*10**9); r['finished_utc']=r['lifetime']['end']['utc']
        r['elapsed_seconds_including_cleanup']=end-r['lifetime']['start']['monotonic_before_ns']/1e9
    end_receipt(lf.supervisor['worker'],1003); lf.supervisor['worker']['timeout_seconds']=950
    end_receipt(lf.telemetry['cpu_receipt'],997); lf.telemetry['cpu_receipt']['timeout_seconds']=930
    lf.telemetry['lifetime']['end']=processes.stamp(1000*10**9)
    lf.telemetry['gpu_stop_clock']=processes.stamp(998*10**9)
    with patch.object(telemetry,'BASE',datetime.fromisoformat(processes.stamp(96*10**9)['utc'])):
        cpu_rows=[json.loads(line) for line in telemetry.cpu_fixture().splitlines()]
        gpu=telemetry.gpu_fixture()
    for row in cpu_rows[1:-1]:
        row['monotonic_start_ns']-=904*10**9; row['monotonic_end_ns']-=904*10**9
    cpu=telemetry.cpu_bytes(cpu_rows)
    lf.ready.update(cpu_metadata=cpu_rows[0],first_gpu_row=list(csv.reader(io.StringIO(gpu.decode())))[1],
                    local_time_with_offset=processes.stamp(97*10**9)['utc'])
    lf.collection['ready_canonical_sha256']=digest(lf.ready)
    lf.telemetry['gpu_sha256']=hashlib.sha256(gpu).hexdigest()
    receipts={'restorer':lf.outer,'set':lf.restoration['set'],'worker':lf.restoration['worker'],
              'reset':lf.restoration['reset_attempts'][0],'telemetry_worker':lf.supervisor['worker'],
              'cpu':lf.telemetry['cpu_receipt']}
    roles={}
    for name,r in receipts.items():
        r.update(argv=[identities['python']['path'],'-S',name],cwd=str(root),capture_sha256=hashlib.sha256(b'').hexdigest())
        if name=='cpu':
            r['argv'].append(str(directory/'telemetry/cpu.jsonl')); r['report_limit_bytes']=REPORT_BYTES
        if name in ('set','worker','reset'): lf.obligation[name+'_argv']=r['argv']
        roles[name]={k:r[k] for k in FIELDS}
    inputs=dict(schema_version=1,kind='clock-finalization-inputs',collection_root=str(directory),jobs=jobs,roles=roles,
        library_contract=contract,injection=identities['injection.so'],frozen_identities=list(identities.values()),
        path_bindings=[],library_roots=[str(frozen)],sources=sources,python_metadata=metadata,cpu_source=telemetry.SOURCE,
        gpu_uuid=telemetry.UUID,timezone_name='UTC',cwd=str(root),external_cancel_check_enabled=False)
    plan=dict(schema_version=1,diagnostic_only=True,jobs=jobs,work_deadline_monotonic_seconds=691,
        capture_limit_bytes_per_job=CAPTURE_BYTES,report_limit_bytes_per_job=REPORT_BYTES,
        minimum_initial_free_bytes=MIN_FREE_BYTES,python_copies_deferred=True,launch_readiness_verified=False)
    def report_receipt(path,raw):
        write(path,raw); identity=path.stat()
        return dict(path=str(path),size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            device=identity.st_dev,inode=identity.st_ino,post_exit_hash_verified=True,producer_identity_verified=False)
    def writer(mode=None):
        if mode=='runner': raise RuntimeError('failed runner')
        directory.mkdir(mode=0o700)
        if mode=='telemetry': cpu_rows[-1]['actual_elapsed_seconds']=899
        current_cpu=telemetry.cpu_bytes(cpu_rows)
        receipts['cpu']['report_capture']=report_receipt(directory/'telemetry/cpu.jsonl',current_cpu)
        write(directory/'telemetry/gpu.csv',gpu)
        if mode=='lifecycle': lf.collection['observer_shutdown']['kill_escalation']=True
        if mode=='restoration': lf.restoration['reset_request_within_deadline']=False
        for index,(job,row,raw) in enumerate(zip(jobs,sequence['jobs'],raw_reports)):
            path=Path(job['report'])
            if mode=='output' and index==17:
                bad=json.loads(raw); bad['warmups']=False; raw=json.dumps(bad).encode()
            if mode=='process' and index==17: row['observation']['samples'][0]['affinity']=[0]
            row['receipt']['report_capture']=report_receipt(path,raw)
            write(path.with_suffix('.capture'),b'')
            record(path.with_suffix('.receipt.json'),row['receipt']); record(path.with_suffix('.process.json'),row['observation'])
        record(directory/'sequence/sequence-plan.json',plan); record(directory/'sequence/sequence-result.json',sequence)
        for name,prefix in (('restorer','restorer'),('set','restoration/set'),('worker','restoration/worker'),
                           ('reset','restoration/reset-1'),('telemetry_worker','observer-supervision/worker'),('cpu','telemetry/cpu-command')):
            record(directory/(prefix+'.receipt.json'),receipts[name]); write(directory/(prefix+('.capture' if name=='restorer' else '.txt')),b'')
        for name,value in (('collection.json',lf.collection),('restoration/restoration.json',lf.restoration),
            ('restoration/obligation.json',lf.obligation),('observer-supervision/supervisor.json',lf.supervisor),
            ('telemetry/telemetry.json',lf.telemetry),('telemetry/ready.json',lf.ready)):
            record(directory/name,value)
        write(directory/'observer-worker.txt',b'')
        if mode=='capture': write(directory/'restorer.capture',b'changed')
        if mode=='frozen': write(frozen/'optional.so',b'\x7fELFwrong!!!')
        returned=deepcopy(lf.collection)
        if mode=='return': returned['status']='failed'
        if mode=='input_copy': write(root/'launch/finalizer-inputs.json',b'changed')
        return dict(collection=returned,work_deadline=692 if mode=='deadline' else 691)
    return inputs,writer


def exercise(case):
    archive=archived_inputs()
    modes={'success':[None], 'gates':['runner','return','deadline','input_copy','capture','telemetry','lifecycle',
            'restoration','process','output','frozen','late_evidence','gate_binding','gate_clock','gate_flag'], 'publication':['publish','marker']}[case]
    for mode in modes:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); inputs,writer=data(root,archive); raw=json.dumps(inputs).encode()
            sha=hashlib.sha256(raw).hexdigest(); events=[]
            def run(): events.append('runner'); return writer(mode)
            actual_qualify=finalizer.qualify; actual_publish=finalizer.publish
            def qualification(*a,**kw):
                value=actual_qualify(*a,**kw)
                if mode=='late_evidence': write(root/'collection/restorer.capture',b'late mutation')
                return value
            def publication(directory,name,raw):
                if (mode=='publish' and name=='output-qualification.json') or (mode=='marker' and name=='completion.pending.json'):
                    # Model a failed/short publication, retaining its partial bytes.
                    write(directory/name,b'partial'); raise OSError('test publication failure')
                return actual_publish(directory,name,raw)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for name in ('read_sequence','read_control','validate_telemetry','validate_lifecycle','validate_processes','qualify','verify_files'):
                    original=qualification if name=='qualify' else getattr(finalizer,name)
                    def call(*a,_name=name,_original=original,**kw):
                        events.append(_name); value=_original(*a,**kw)
                        if mode=='gate_binding' and _name=='qualify': value['jobs'][-1]['receipt_canonical_sha256']='0'*64
                        if mode=='gate_clock' and _name=='validate_processes':
                            value['clock_domain_offset_ns']={'minimum':0,'maximum':1}
                        if mode=='gate_flag' and _name=='validate_telemetry': value['telemetry_contents_validated']=False
                        return value
                    stack.enter_context(patch.object(finalizer,name,side_effect=call))
                stack.enter_context(patch.object(finalizer,'publish',side_effect=publication))
                if mode is None:
                    result=finalizer.run_then_finalize(raw,sha,root/'launch',run)
                    assert events==['runner','read_sequence','read_control','validate_telemetry','validate_lifecycle',
                                    'validate_processes','qualify','verify_files'],events
                    assert len(result['consumed_raw_files'])==95 and len(result['results'])==6
                    assert result['validation_composition_complete'] and result['frozen_rehash_after_runner_and_validation']
                    assert not any(result[k] for k in ('launch_provenance_verified','runner_reaping_independently_verified',
                        'hardware_policy_restoration_verified','clock_enforcement_verified','performance_acceptance_evaluated',
                        'complete_preparation_verified','operator_trace_parity_verified'))
                    for name,identity in result['results'].items():
                        path=root/'launch/completion'/name
                        assert len(path.read_bytes())==identity['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==identity['sha256']
                    assert json.loads((root/'launch/completion/completion.json').read_bytes())==result
                    assert (root/'launch/completion/completion.json').stat().st_mode & 0o077 == 0
                else:
                    try: finalizer.run_then_finalize(raw,sha,root/'launch',run)
                    except (ValueError,OSError,RuntimeError): pass
                    else: raise AssertionError('failed gate accepted: '+mode)
                    assert not (root/'launch/completion/completion.json').exists(),mode
                    stages=['runner','read_sequence','read_control','validate_telemetry','validate_lifecycle',
                            'validate_processes','qualify','verify_files']
                    terminal={'runner':0,'input_copy':0,'deadline':1,'return':2,'capture':2,'telemetry':3,
                        'gate_flag':3,'lifecycle':4,'restoration':4,'process':5,'output':6,'late_evidence':6,
                        'gate_binding':6,'gate_clock':6,'frozen':7,'publish':7,'marker':7}[mode]
                    assert events==stages[:terminal+1],(mode,events)
                    assert (root/'launch/finalizer-inputs.json').exists()
                assert events.count('runner')==1
                try: finalizer.run_then_finalize(raw,sha,root/'launch',run)
                except ValueError: pass
                else: raise AssertionError('attempt reused')
                assert events.count('runner')==1
    print(case,'passed',len(modes),'cases; actual validators, synthetic collection only')


class ClockFinalizationTests(unittest.TestCase):
    def child(self,case):
        result=subprocess.run([sys.executable,'-S','-c',
            'from test_finalize_clock_diagnostic import exercise; exercise('+repr(case)+')'],
            cwd=Path(__file__).parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout.decode(errors='replace'))

    def test_real_gate_composition_and_publication(self): self.child('success')
    def test_failed_gates_preserve_partial_evidence_without_rehash_or_completion(self): self.child('gates')
    def test_failed_publication_never_exposes_completion_marker(self): self.child('publication')

    def test_bad_inputs_never_invoke_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            for raw,sha in ((b'{}','0'*64),(b'{}',hashlib.sha256(b'{}').hexdigest()),
                            (b'{"x":1,"x":2}',hashlib.sha256(b'{"x":1,"x":2}').hexdigest())):
                with patch('subprocess.Popen',side_effect=AssertionError('no child')), self.assertRaises(ValueError):
                    finalizer.run_then_finalize(raw,sha,Path(tmp)/'launch',lambda: self.fail('runner called'))
            self.assertEqual(list(Path(tmp).iterdir()),[])


if __name__ == '__main__': unittest.main()
