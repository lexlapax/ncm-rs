"""Temporary control files plus actual short sibling fake-policy transport."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clock_evidence import Evidence, read_sequence
from clock_control_evidence import read_control, ROLES
from validate_clock_lifecycle import validate as validate_lifecycle
from test_clock_evidence import write, record
import test_clock_evidence as sequence_fixture
import test_clock_diagnostic_coordinator as live


def install(root):
    for name in ('restoration','observer-supervision','telemetry'): (root/name).mkdir(mode=0o700)
    roles={}; receipts={}
    for i,name in enumerate(ROLES):
        roles[name]=dict(argv=['/python','-S',name],cwd=str(root),timeout_seconds=10,
            termination_grace_seconds=60 if name=='restorer' else 0,output_limit_bytes=1024**2,
            report_limit_bytes=16*1024**2 if name=='cpu' else None)
        receipts[name]=dict(schema_version=1,status='complete',issues=[],returncode=0,direct_child_reaped=True,
            capture_eof=True,group_kill='sent',interrupt_signal=None,pid=100+i,**deepcopy(roles[name]),
            termination_signal='not_attempted',termination_grace_exhausted=False,process_observer_enabled=False,
            workload_output_qualified=False,executable_identity_verified=False,captured_bytes=0,observed_bytes=0,
            capture_sha256=hashlib.sha256(b'').hexdigest(),report_capture=None)
    cpu=root/'telemetry/cpu.jsonl'; write(cpu,b'{"fixture":true}\n'); current=cpu.stat()
    receipts['cpu']['report_capture']=dict(path=str(cpu),size_bytes=current.st_size,device=current.st_dev,inode=current.st_ino,
        sha256=hashlib.sha256(cpu.read_bytes()).hexdigest(),post_exit_hash_verified=True,producer_identity_verified=False)
    write(root/'telemetry/gpu.csv',b'fake,gpu\n')
    objects={'collection.json':dict(schema_version=1,status='complete'),
        'restoration/restoration.json':dict(schema_version=2,status='complete',set=receipts['set'],
            worker=receipts['worker'],reset_attempts=[receipts['reset']]),
        'restoration/obligation.json':dict(schema_version=2,armed_before_set=True,
            **{k+'_argv':roles[k]['argv'] for k in ('set','worker','reset')}),
        'observer-supervision/supervisor.json':dict(schema_version=1,status='complete',worker=receipts['telemetry_worker']),
        'telemetry/telemetry.json':dict(schema_version=1,status='complete',cpu_receipt=receipts['cpu'],
            gpu_sha256=hashlib.sha256(b'fake,gpu\n').hexdigest()),
        'telemetry/ready.json':dict(schema_version=1,kind='telemetry_ready')}
    for name,prefix in (('restorer','restorer'),('set','restoration/set'),('worker','restoration/worker'),
                        ('reset','restoration/reset-1'),('telemetry_worker','observer-supervision/worker'),
                        ('cpu','telemetry/cpu-command')):
        record(root/(prefix+'.receipt.json'),receipts[name])
        write(root/(prefix+('.capture' if name=='restorer' else '.txt')),b'')
    write(root/'observer-worker.txt',b'post-only fixture stdout\n')
    for name,value in objects.items(): record(root/name,value)
    return roles,objects,receipts


class ControlEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.roles,self.objects,self.receipts=install(self.root)

    def read(self):
        with Evidence(self.root) as reader:
            value=read_control(reader,roles=self.roles)
            return value,deepcopy(reader.inputs)

    def save(self):
        for name,value in self.objects.items(): record(self.root/name,value)

    def test_21_control_files_and_explicit_post_only_stdout(self):
        with patch('subprocess.Popen',side_effect=AssertionError('no processes')):
            value,files=self.read()
        self.assertEqual(len(files),21); self.assertEqual(len(value['binding']['roles']),6)
        self.assertTrue(value['binding']['control_file_receipt_consistency_verified'])
        self.assertEqual(value['binding']['observer_supervisor_capture']['binding'],'post_only')
        self.assertFalse(value['binding']['observer_supervisor_capture']['producer_digest_verified'])
        self.assertTrue(all(not r['producer_capture_inode_verified'] for r in value['binding']['roles']))
        for flag in ('launch_provenance_verified','lifecycle_qualified','telemetry_contents_qualified',
                     'hardware_policy_restoration_verified','postflight_call_order_enforced'):
            self.assertFalse(value['binding'][flag])
        self.assertEqual(value['cpu_raw'],b'{"fixture":true}\n')

    def test_all_95_control_and_sequence_files_share_one_reader_budget(self):
        fixture=sequence_fixture.ClockEvidenceTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        roles,_,_=install(fixture.root)
        with Evidence(fixture.root) as reader:
            sequence=read_sequence(reader,jobs=fixture.jobs,work_deadline=fixture.deadline)
            control=read_control(reader,roles=roles)
            self.assertEqual(len(reader.inputs),95)
            self.assertEqual(len(sequence['raw_reports']),18)
            self.assertTrue(control['binding']['control_file_receipt_consistency_verified'])
            reader.verify_unchanged()

    def test_role_declarations_reject_before_file_reads(self):
        for mode in ('missing','unknown','field','relative','cwd','boolean','limit','report'):
            roles=deepcopy(self.roles)
            if mode=='missing': roles.pop('cpu')
            if mode=='unknown': roles['unknown']=roles['cpu']
            if mode=='field': roles['worker']['unknown']=1
            if mode=='relative': roles['worker']['argv'][0]='python'
            if mode=='cwd': roles['worker']['cwd']='relative'
            if mode=='boolean': roles['cpu']['timeout_seconds']=True
            if mode=='limit': roles['cpu']['output_limit_bytes']=2*1024**2
            if mode=='report': roles['cpu']['report_limit_bytes']=None
            with self.subTest(mode=mode), Evidence(self.root) as reader, \
                 patch.object(reader,'record',side_effect=AssertionError('no reads before complete declarations')):
                with self.assertRaises(ValueError): read_control(reader,roles=roles)

    def test_failed_records_embedded_receipts_commands_and_captures(self):
        for mode in ('status','retry','obligation','embedded','argv','cwd','budget','pid','termination','bytes','scope','capture','gpu','stdout_limit'):
            fixture=ControlEvidenceTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
            if mode=='status': fixture.objects['collection.json']['status']='failed'
            if mode=='retry': fixture.objects['restoration/restoration.json']['reset_attempts'].append(deepcopy(fixture.receipts['reset']))
            if mode=='obligation': fixture.objects['restoration/obligation.json']['set_argv']=['/wrong']
            if mode=='embedded': fixture.receipts['set']['pid']=999  # Embedded changes, individual file does not.
            if mode=='argv': fixture.roles['cpu']['argv']=['/wrong']
            if mode=='cwd': fixture.roles['cpu']['cwd']='/different'
            if mode=='budget': fixture.roles['cpu']['timeout_seconds']=11
            receipt=fixture.receipts['cpu']
            if mode=='pid': receipt['pid']=100
            if mode=='termination': receipt['termination_signal']='sent'
            if mode=='bytes': receipt['observed_bytes']=1
            if mode=='scope': receipt['executable_identity_verified']=True
            if mode=='capture': write(fixture.root/'restorer.capture',b'changed')
            if mode=='gpu': write(fixture.root/'telemetry/gpu.csv',b'changed')
            if mode=='stdout_limit': write(fixture.root/'observer-worker.txt',b'x'*(1024**2+1))
            record(fixture.root/'telemetry/cpu-command.receipt.json',receipt); fixture.save()
            with self.subTest(mode=mode), self.assertRaises(ValueError): fixture.read()

    def test_cpu_same_bytes_new_inode_and_metadata_changed_after_read(self):
        path=self.root/'telemetry/cpu.jsonl'; raw=path.read_bytes()
        path.rename(path.with_suffix('.saved')); write(path,raw)
        with self.assertRaisesRegex(ValueError,'CPU telemetry report file'): self.read()
        path.unlink(); path.with_suffix('.saved').rename(path)
        actual=Evidence.raw
        def later(reader,name,*args):
            data=actual(reader,name,*args)
            if name=='observer-worker.txt': write(reader.root/'restorer.capture',b'changed')
            return data
        with patch.object(Evidence,'raw',later), self.assertRaisesRegex(ValueError,'after read'): self.read()

    def test_real_predeclared_roles_and_control_file_writer_reader(self):
        if getattr(live.ClockCoordinatorTests,'__unittest_skip__',False): self.skipTest('Linux CPU0/2 and taskset required')
        fixture=live.ClockCoordinatorTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        mask=os.umask(0o077)
        try: root,_=fixture.run_case('lifecycle_validation')
        finally: os.umask(mask)
        roles=json.loads((root/'declared-roles.json').read_bytes())
        with Evidence(root/'collection') as reader:
            value=read_control(reader,roles=roles)
            self.assertEqual(len(reader.inputs),21)
            sequence=reader.record('sequence/sequence-result.json')
            samples=[json.loads(line) for line in value['cpu_raw'].splitlines()]
            deadline=json.loads((root/'received-deadline.json').read_bytes())['deadline']
            lifecycle=validate_lifecycle(value['collection'],value['restorer_receipt'],value['restoration'],
                value['obligation'],sequence,value['observer_supervisor'],value['telemetry'],value['ready'],
                [row for row in samples if row['kind']=='sample'],
                commands={k:roles[k]['argv'] for k in ('set','worker','reset')},cwd=str(live.HERE),
                work_deadline=deadline,external_cancel_check_enabled=False)
            self.assertEqual(len(lifecycle['jobs']),18)
            reader.verify_unchanged()
        self.assertTrue(all(not Path('/proc/'+str(row['pid'])).exists() for row in value['binding']['roles']))
        self.assertFalse(value['binding']['telemetry_contents_qualified'])


if __name__ == '__main__': unittest.main()
