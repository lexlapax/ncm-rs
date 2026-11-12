"""Independent negative evidence tests plus real CPU-only writer/reader transport."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import time
import unittest
from unittest.mock import patch

from clock_process_observation import ORDER
from validate_readonly_lifecycle import digest
from verify_clock_processes import validate_processes
import test_clock_process_observation as live
import clock_codec_sequence as sequence


def image(path, inode):
    return dict(path=path,device=29,inode=inode,bytes=100,sha256='a'*64)


def mapped(value):
    return dict(path=value['path'],device=28,inode=value['inode'],path_device=29,
                path_inode=value['inode'],kernel_map_link_verified=True)


def stamp(ns):
    return dict(monotonic_before_ns=ns,monotonic_after_ns=ns,
                utc=(datetime(2026,11,12,tzinfo=timezone.utc)+timedelta(microseconds=ns//1000)).isoformat())


def lifetime(start,end):
    return dict(pid=42,parent_pid=41,start=stamp(start),end=stamp(end))


def rehash(observation):
    translated = {old:digest(value) for old,value in observation['mapping_sets'].items()}
    observation['mapping_sets'] = {translated[k]:v for k,v in observation['mapping_sets'].items()}
    for sample in observation['samples']: sample['mapping_sha256'] = translated[sample['mapping_sha256']]


class ClockProcessValidationTests(unittest.TestCase):
    def setUp(self):
        self.injection = image('/injection.so',99)
        self.required = image('/lib/required.so',2)
        self.optional = image('/lib/optional.so',3)
        self.lazy = image('/lib/lazy.so',4)
        self.contract = dict(schema_version=1,prepared_declaration_consistency_verified=True,
                             runtime_mapping_verified=False,launch_ready=False,taskset=image('/taskset',1),
                             workloads={r:dict(command_path='/'+r,image=image('/'+r,10+i))
                                        for i,r in enumerate(('previous','python','candidate'))},
                             libraries={r:[self.required] for r in ('previous','python','candidate')},
                             known_libraries=[self.required,self.optional])
        self.jobs=[]; rows=[]
        for i,runtime in enumerate(ORDER):
            ordinal=i+1; start=(101+i)*10**9; end=start+500_000_000
            job=dict(runtime=runtime,repeat=i//3+1,workload='codec',cwd='/checkout',
                     report=f'/reports/{ordinal}.json',argv=['/taskset','-c','2','/'+runtime,f'/reports/{ordinal}.json'])
            self.jobs.append(job)
            mapping=dict(files=sorted([mapped(self.contract['workloads'][runtime]['image']),
                                       mapped(self.required),mapped(self.optional),mapped(self.lazy)],key=lambda v:v['path']),
                         non_file_executable_mappings=['[vdso]'])
            observed=dict(schema_version=1,status='complete',issues=[],observation_failures=[],pid=1000+i,
                          process_start_ticks=500+i,transient_samples_discarded=0,startup_incomplete_samples=0,
                          proc_unavailable_samples=0,startup_permission_denied_samples=0,
                          samples=[dict(monotonic_start_ns=start+200_000_000,monotonic_end_ns=start+201_000_000,
                                        phase='workload',affinity=[2],injection_observed=False,mapping_sha256=digest(mapping))],
                          mapping_sets={digest(mapping):mapping})
            for k in ('launch_provenance_verified','whole_lifetime_mapping_coverage','mapped_file_bytes_verified',
                      'environment_values_serialized'): observed[k]=False
            receipt=dict(schema_version=1,status='complete',issues=[],returncode=0,direct_child_reaped=True,
                         capture_eof=True,group_kill='sent',interrupt_signal=None,pid=1000+i,argv=job['argv'],cwd=job['cwd'],
                         process_observer_enabled=True,output_limit_bytes=1024**2,report_limit_bytes=16*1024**2,
                         timeout_seconds=590,termination_grace_seconds=0,termination_grace_exhausted=False,
                         termination_signal='not_attempted',workload_output_qualified=False,executable_identity_verified=False,
                         captured_bytes=0,observed_bytes=0,capture_sha256='b'*64,spawn_returned_monotonic_seconds=start/1e9+.001,
                         lifetime=lifetime(start,end),started_utc=stamp(start)['utc'],finished_utc=stamp(end)['utc'],
                         elapsed_seconds_including_cleanup=.5,
                         report_capture=dict(path=job['report'],size_bytes=1,sha256='c'*64,device=29,inode=10000+i,
                                             post_exit_hash_verified=True,producer_identity_verified=False))
            rows.append(dict(ordinal=ordinal,runtime=runtime,repeat=i//3+1,receipt=receipt,observation=observed))
        self.deadline=130.0
        self.result=dict(schema_version=1,status='complete',stop_reason=None,external_cancelled=False,diagnostic_only=True,
                         python_copies_deferred=True,launch_provenance_verified=False,benchmark_outputs_qualified=False,
                         whole_run_supervision_provided=False,work_deadline_monotonic_seconds=self.deadline,
                         lifetime=lifetime(100*10**9,120*10**9),jobs=rows)

    def validate(self, result=None, contract=None):
        return validate_processes(self.jobs,result or self.result,contract or self.contract,
                                  injection=self.injection,work_deadline=self.deadline)

    def test_complete_metadata_scope_and_optional_absence_without_io(self):
        with patch('builtins.open',side_effect=AssertionError('no file IO')), \
             patch.object(Path,'open',side_effect=AssertionError('no file IO')), \
             patch('subprocess.Popen',side_effect=AssertionError('no processes')):
            checked=self.validate()
        self.assertEqual(len(checked['jobs']),18)
        self.assertEqual(checked['known_paths_observed'],['/lib/optional.so','/lib/required.so'])
        self.assertEqual(checked['unlisted_paths_observed'], ['/lib/lazy.so'])
        self.assertIn('/python',checked['prepared_paths_observed'])
        self.assertTrue(checked['process_receipt_consistency_verified'])
        for k in ('launch_provenance_verified','mapped_file_bytes_verified','restoration_verified',
                  'telemetry_coverage_verified','benchmark_outputs_qualified','frozen_files_rechecked'):
            self.assertFalse(checked[k])
        for row in self.result['jobs']:
            o=row['observation']
            for m in o['mapping_sets'].values(): m['files']=[v for v in m['files'] if v['path']!=self.optional['path']]
            rehash(o)
        self.assertNotIn(self.optional['path'],self.validate()['known_paths_observed'])

    def test_rehashed_bad_maps_cannot_hide_known_required_image_or_injection_failures(self):
        for mode in ('known','required','image','injection','link','kernel_inode','duplicate','order','unused'):
            with self.subTest(mode=mode):
                result=deepcopy(self.result); observed=result['jobs'][0]['observation']
                mapping=next(iter(observed['mapping_sets'].values())); files=mapping['files']
                if mode=='known':
                    v=next(v for v in files if v['path']==self.optional['path']); v['inode']+=100; v['path_inode']+=100
                if mode=='required': mapping['files']=[v for v in files if v['path']!=self.required['path']]
                if mode=='image': mapping['files']=[v for v in files if v['path']!='/previous']
                if mode=='injection': mapping['files']=sorted([*files,mapped(self.injection)],key=lambda v:v['path'])
                if mode=='link': files[0]['kernel_map_link_verified']=False
                if mode=='kernel_inode': files[0]['inode']+=1
                if mode=='duplicate': files.append(deepcopy(files[0]))
                if mode=='order': files.reverse()
                if mode=='unused':
                    extra=deepcopy(mapping); extra['non_file_executable_mappings'].append('[extra]')
                    observed['mapping_sets'][digest(extra)]=extra
                rehash(observed)
                with self.assertRaises(ValueError): self.validate(result)

    def test_record_order_pid_commands_report_and_scope_failures(self):
        for mode in ('prefix','order','pid','self_pid','command','report','cap','claim','observer','cancel','schema','failed','digest','grace'):
            with self.subTest(mode=mode):
                result=deepcopy(self.result); row=result['jobs'][-1]; receipt=row['receipt']; observed=row['observation']
                if mode=='prefix': result['jobs'].pop()
                if mode=='order': row['repeat']=True
                if mode=='pid': receipt['pid']=observed['pid']=1000
                if mode=='self_pid': receipt['pid']=observed['pid']=42
                if mode=='command': receipt['argv']=['unexpected-not-for-output']
                if mode=='report': receipt['report_capture']['path']='/other.json'
                if mode=='cap': receipt['observed_bytes']=1
                if mode=='claim': observed['mapped_file_bytes_verified']=True
                if mode=='observer': receipt['process_observer_enabled']=False
                if mode=='cancel': result['external_cancelled']=True
                if mode=='schema': observed['schema_version']=True
                if mode=='failed': observed['observation_failures']=['original failure']
                if mode=='digest': receipt['capture_sha256']='wrong'
                if mode=='grace': receipt['termination_grace_seconds']=False
                with self.assertRaises(ValueError): self.validate(result)

    def test_deadline_clock_envelope_affinity_and_sampling_failures(self):
        for mode in ('deadline','late','spawn','parent','overlap','sample','affinity','phase','empty','counter','clock','interval'):
            with self.subTest(mode=mode):
                result=deepcopy(self.result); row=result['jobs'][-1]; receipt=row['receipt']; observed=row['observation']
                if mode=='deadline': result['work_deadline_monotonic_seconds']+=1
                if mode=='late': result['lifetime']['end']=stamp(131*10**9)
                if mode=='spawn': receipt['spawn_returned_monotonic_seconds']=float('nan')
                if mode=='parent': receipt['lifetime']['parent_pid']+=2
                if mode=='overlap':
                    first=result['jobs'][0]
                    for k in ('lifetime','started_utc','finished_utc','elapsed_seconds_including_cleanup','spawn_returned_monotonic_seconds'):
                        receipt[k]=deepcopy(first['receipt'][k])
                    for k in ('monotonic_start_ns','monotonic_end_ns'):
                        observed['samples'][0][k]=first['observation']['samples'][0][k]
                if mode=='sample': observed['samples'][0]['monotonic_end_ns']=200*10**9
                if mode=='affinity': observed['samples'][0]['affinity']=[0]
                if mode=='phase': observed['samples'][0]['phase']='unknown'
                if mode=='empty': observed['samples']=[]
                if mode=='counter': observed['startup_permission_denied_samples']=1
                if mode=='clock': receipt['lifetime']['start']['utc']='2000-01-01T00:00:00+00:00'
                if mode=='interval': observed['samples'].append(deepcopy(observed['samples'][0]))
                with self.assertRaises(ValueError): self.validate(result)

    def test_contract_conflicts_cannot_downgrade_known_path_to_post_only(self):
        contract=deepcopy(self.contract); contract['known_libraries'].pop(0)
        with self.assertRaises(ValueError): self.validate(contract=contract)
        contract=deepcopy(self.contract); contract['known_libraries'].append(deepcopy(contract['known_libraries'][0]))
        with self.assertRaises(ValueError): self.validate(contract=contract)
        contract=deepcopy(self.contract); contract['known_libraries'][1]['inode']+=1
        with self.assertRaises(ValueError): self.validate(contract=contract)
        for path in (self.jobs[0]['report'], '/bad/../report.json'):
            with self.subTest(path=path):
                jobs,result=deepcopy((self.jobs,self.result))
                jobs[-1]['report']=path; jobs[-1]['argv'][-1]=path
                result['jobs'][-1]['receipt']['report_capture']['path']=path
                with self.assertRaises(ValueError):
                    validate_processes(jobs,result,self.contract,injection=self.injection,work_deadline=self.deadline)

    def test_actual_cpu_sequence_writer_reader_integration(self):
        if getattr(live.ClockObservationTests,'__unittest_skip__',False): self.skipTest('Linux CPU2/taskset required')
        fixture=live.ClockObservationTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        known=[*fixture.libraries['previous'],fixture.injection]
        contract=dict(schema_version=1,prepared_declaration_consistency_verified=True,runtime_mapping_verified=False,
                      launch_ready=False,workloads=fixture.workloads,taskset=fixture.taskset,
                      libraries=fixture.libraries,known_libraries=known)
        deadline=time.monotonic()+30
        with patch.object(sequence,'plan_jobs',return_value=fixture.jobs):
            result=sequence.run_sequence(fixture.previous,fixture.binaries,fixture.output,root=fixture.root,env=fixture.env,
                                         work_deadline=deadline,cancel_check=lambda:False,
                                         process_observer_factory=fixture.factory(known_libraries=known),
                                         taskset=fixture.taskset['path'])
        saved=json.loads((fixture.output/'sequence-result.json').read_bytes())
        plan=json.loads((fixture.output/'sequence-plan.json').read_bytes())['jobs']
        self.assertEqual(saved,result); self.assertEqual(plan,fixture.jobs)
        for job,row in zip(plan,saved['jobs']):
            prefix=Path(job['report'])
            self.assertEqual(row['observation'],json.loads(prefix.with_suffix('.process.json').read_bytes()))
            self.assertEqual(row['receipt'],json.loads(prefix.with_suffix('.receipt.json').read_bytes()))
        checked=validate_processes(plan,saved,contract,injection=fixture.injection,work_deadline=deadline)
        self.assertEqual(len(checked['jobs']),18)
        self.assertTrue(all(not Path('/proc/'+str(v['pid'])).exists() for v in checked['jobs']))
        self.assertFalse(checked['benchmark_outputs_qualified'])


if __name__ == '__main__': unittest.main()
