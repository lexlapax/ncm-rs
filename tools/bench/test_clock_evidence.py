"""Private temporary evidence, actual CPU transport, and isolated archived outputs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import clock_evidence as evidence
import test_clock_process_observation as live
from clock_process_observation import ORDER
from clock_codec_sequence import CAPTURE_BYTES, REPORT_BYTES, MIN_FREE_BYTES


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    path.write_bytes(raw); path.chmod(0o600)


def record(path,value): write(path,json.dumps(value).encode())


class ClockEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.deadline=1000.
        (self.root/'sequence').mkdir(mode=0o700)
        self.jobs=[]; rows=[]
        for i,runtime in enumerate(ORDER):
            repeat=i//3+1; kind='python' if runtime=='python' else 'rust'
            report=self.root/f'sequence/{runtime}/codec-{kind}-{repeat}.json'
            job=dict(runtime=runtime,repeat=repeat,workload='codec',report=str(report),
                     argv=['/taskset','-c','2','/fake',str(report)],cwd=str(self.root))
            self.jobs.append(job)
            raw=json.dumps({'ordinal':i+1}).encode(); capture=f'capture {i+1}'.encode()
            write(report,raw); write(report.with_suffix('.capture'),capture)
            current=report.stat()
            receipt=dict(schema_version=1,status='complete',issues=[],returncode=0,pid=1000+i,
                direct_child_reaped=True,capture_eof=True,group_kill='sent',interrupt_signal=None,
                argv=job['argv'],cwd=job['cwd'],process_observer_enabled=True,output_limit_bytes=CAPTURE_BYTES,
                report_limit_bytes=REPORT_BYTES,captured_bytes=len(capture),observed_bytes=len(capture),
                capture_sha256=hashlib.sha256(capture).hexdigest(),
                report_capture=dict(path=str(report),device=current.st_dev,inode=current.st_ino,
                    size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),post_exit_hash_verified=True,
                    producer_identity_verified=False))
            observation=dict(schema_version=1,status='complete',pid=1000+i)
            record(report.with_suffix('.receipt.json'),receipt); record(report.with_suffix('.process.json'),observation)
            rows.append(dict(ordinal=i+1,runtime=runtime,repeat=repeat,receipt=receipt,observation=observation))
        self.plan=dict(schema_version=1,diagnostic_only=True,jobs=self.jobs,work_deadline_monotonic_seconds=self.deadline,
            capture_limit_bytes_per_job=CAPTURE_BYTES,report_limit_bytes_per_job=REPORT_BYTES,
            minimum_initial_free_bytes=MIN_FREE_BYTES,python_copies_deferred=True,launch_readiness_verified=False)
        self.sequence=dict(schema_version=1,status='complete',stop_reason=None,external_cancelled=False,
            diagnostic_only=True,jobs=rows,work_deadline_monotonic_seconds=self.deadline,python_copies_deferred=True,
            launch_provenance_verified=False,benchmark_outputs_qualified=False,whole_run_supervision_provided=False)
        self.save()

    def save(self):
        record(self.root/'sequence/sequence-plan.json',self.plan)
        record(self.root/'sequence/sequence-result.json',self.sequence)

    def read(self):
        with evidence.Evidence(self.root) as reader:
            result=evidence.read_sequence(reader,jobs=self.jobs,work_deadline=self.deadline)
            return result,deepcopy(reader.inputs)

    def test_all_74_files_bound_without_spawning_or_qualification(self):
        # Captures have producer hash/length, not a producer inode. A same-byte
        # replacement BEFORE the read cannot be rejected as identity drift.
        capture=Path(self.jobs[0]['report']).with_suffix('.capture')
        raw=capture.read_bytes(); old_inode=capture.stat().st_ino
        capture.rename(capture.with_suffix('.saved')); write(capture,raw)
        self.assertNotEqual(capture.stat().st_ino,old_inode)
        with patch('subprocess.Popen',side_effect=AssertionError('no child processes')):
            value,files=self.read()
        self.assertEqual(len(files),74); self.assertEqual(len(value['raw_reports']),18)
        self.assertEqual(value['sequence'],self.sequence)
        self.assertTrue(value['binding']['sequence_file_receipt_consistency_verified'])
        for key in ('launch_provenance_verified','benchmark_outputs_qualified',
                    'capture_prelaunch_inode_verified','postflight_call_order_enforced'):
            self.assertFalse(value['binding'][key])
        self.assertTrue(all(len(v['sha256'])==64 for v in files.values()))

    def test_plan_receipts_observations_and_file_bytes_cannot_disagree(self):
        for mode in ('plan','deadline','prefix','ordinal','command','observation','capture','report','replacement','pid'):
            fixture=ClockEvidenceTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
            row=fixture.sequence['jobs'][-1]; path=Path(fixture.jobs[-1]['report'])
            if mode=='plan': fixture.plan=deepcopy(fixture.plan); fixture.plan['jobs'][-1]['argv'][0]='/wrong'
            if mode=='deadline': fixture.plan['work_deadline_monotonic_seconds']+=1
            if mode=='prefix': fixture.sequence['jobs'].pop()
            if mode=='ordinal': row['ordinal']=True
            if mode=='command':
                row['receipt']['argv']=['/wrong']; record(path.with_suffix('.receipt.json'),row['receipt'])
            if mode=='observation': record(path.with_suffix('.process.json'),{'status':'changed'})
            if mode=='capture': write(path.with_suffix('.capture'),b'wrong')
            if mode=='report': write(path,b'changed')
            if mode=='replacement':
                raw=path.read_bytes(); path.rename(path.with_suffix('.saved')); write(path,raw)
            if mode=='pid':
                row['receipt']['pid']=1000; record(path.with_suffix('.receipt.json'),row['receipt'])
            fixture.save()
            with self.subTest(mode=mode), self.assertRaises(ValueError): fixture.read()

    def test_duplicate_json_keys_nonfinite_and_repeat_reads_fail(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            write(self.root/'bad.json',raw)
            with evidence.Evidence(self.root) as reader, self.assertRaises(ValueError): reader.record('bad.json')
        write(self.root/'bad.json',b'{}')
        with evidence.Evidence(self.root) as reader:
            reader.raw('bad.json')
            with self.assertRaises(ValueError): reader.raw('bad.json')

    def test_symlinks_fifo_permissions_root_replacement_and_paths(self):
        write(self.root/'file',b'hello')
        (self.root/'link').symlink_to('file')
        (self.root/'alias').symlink_to('sequence',target_is_directory=True)
        os.mkfifo(self.root/'fifo',0o600)
        for name in ('../file','/file','./file','sequence//x','link','alias/sequence-plan.json','fifo'):
            with self.subTest(name=name), evidence.Evidence(self.root) as reader:
                with self.assertRaises((ValueError,OSError)): reader.raw(name)
        (self.root/'file').chmod(0o644)
        with evidence.Evidence(self.root) as reader, self.assertRaises(ValueError): reader.raw('file')
        (self.root/'sequence').chmod(0o755)
        with evidence.Evidence(self.root) as reader, self.assertRaises(ValueError): reader.raw('sequence/sequence-plan.json')
        (self.root/'sequence').chmod(0o700)
        with evidence.Evidence(self.root) as reader:
            moved=self.root.with_name(self.root.name+'-moved'); self.root.rename(moved)
            try:
                self.root.mkdir(mode=0o700)
                with self.assertRaises(ValueError): reader.raw('file')
            finally:
                self.root.rmdir(); moved.rename(self.root)

    def test_budgets_reject_before_reading_bytes(self):
        write(self.root/'file',b'hello')
        for name,value in (('FILE_LIMIT',0),('TOTAL_BYTES',1)):
            with self.subTest(name=name), patch.object(evidence,name,value), evidence.Evidence(self.root) as reader, \
                 patch.object(evidence.os,'read',side_effect=AssertionError('no bytes should be read')):
                with self.assertRaises(ValueError): reader.raw('file')
        with evidence.Evidence(self.root) as reader, patch.object(evidence.os,'read',side_effect=AssertionError('no bytes')):
            with self.assertRaises(ValueError): reader.raw('file',1)

    def test_replacement_during_read_and_earlier_change_during_later_read(self):
        path=self.root/'file'; write(path,b'hello'); real=evidence.os.read; changed=False
        def replaced(fd,count):
            nonlocal changed
            data=real(fd,count)
            if not changed:
                changed=True; path.rename(self.root/'saved'); write(path,b'hello')
            return data
        with evidence.Evidence(self.root) as reader, patch.object(evidence.os,'read',side_effect=replaced):
            with self.assertRaises(ValueError): reader.raw('file')
        real_raw=evidence.Evidence.raw; victim=Path(self.jobs[0]['report']).with_suffix('.capture')
        def later(reader,name,*args):
            raw=real_raw(reader,name,*args)
            if name==str(Path(self.jobs[-1]['report']).relative_to(self.root)): write(victim,b'changed')
            return raw
        with patch.object(evidence.Evidence,'raw',later), self.assertRaisesRegex(ValueError,'after read'): self.read()

    def test_real_eighteen_cpu_job_writer_reader_transport(self):
        if getattr(live.ClockObservationTests,'__unittest_skip__',False): self.skipTest('Linux CPU2/taskset required')
        fixture=live.ClockObservationTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        deadline=time.monotonic()+30; mask=os.umask(0o077)
        try:
            with patch.object(live.sequence,'plan_jobs',return_value=fixture.jobs):
                result=live.sequence.run_sequence(fixture.previous,fixture.binaries,fixture.output,root=fixture.root,
                    env=fixture.env,work_deadline=deadline,cancel_check=lambda:False,
                    process_observer_factory=fixture.factory(),taskset=fixture.taskset['path'])
        finally: os.umask(mask)
        with evidence.Evidence(fixture.root) as reader:
            read=evidence.read_sequence(reader,jobs=fixture.jobs,work_deadline=deadline,sequence_dir=fixture.output.name)
            self.assertEqual(read['sequence'],result); self.assertEqual(len(reader.inputs),74)
        self.assertTrue(all(not Path('/proc/'+str(row['receipt']['pid'])).exists() for row in result['jobs']))

    def test_archived_reports_through_reader_and_actual_qualifier_in_fresh_child(self):
        child=subprocess.run([sys.executable,'-c','from test_clock_evidence import archived_transport; archived_transport()'],
                             cwd=Path(__file__).resolve().parent,capture_output=True,text=True,timeout=60)
        self.assertEqual(child.returncode,0,child.stdout+child.stderr)
        self.assertEqual(child.stdout.strip(),'18 archived reports / 74 files / output qualification passed')


def archived_transport():
    from test_qualify_clock_codec import archived_inputs
    from qualify_clock_codec import qualify
    raw,_,sources,metadata=archived_inputs()
    fixture=ClockEvidenceTests(); fixture.setUp()
    try:
        for data,job,row in zip(raw,fixture.jobs,fixture.sequence['jobs']):
            path=Path(job['report']); write(path,data); current=path.stat()
            row['receipt']['report_capture'].update(size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                                                    device=current.st_dev,inode=current.st_ino)
            record(path.with_suffix('.receipt.json'),row['receipt'])
        fixture.save(); value,files=fixture.read()
        checked=qualify(value['raw_reports'],value['sequence']['jobs'],sources=sources,python_metadata=metadata)
        assert checked['reported_codec_outputs_qualified'] and len(files)==74
        assert not checked['launch_provenance_verified']
        print('18 archived reports / 74 files / output qualification passed')
    finally: fixture.doCleanups()


if __name__ == '__main__': unittest.main()
