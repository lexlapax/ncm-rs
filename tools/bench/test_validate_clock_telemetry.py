"""Full virtual-clock serializer inputs, not new telemetry collection or GPU work."""
from copy import deepcopy
import csv
from datetime import timedelta
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import test_validate_readonly_telemetry as fixtures
import test_verify_clock_processes as processes
from clock_process_observation import ORDER
from validate_clock_telemetry import validate


def stamp(seconds):
    ns=int((1000+seconds)*1e9)
    return dict(monotonic_before_ns=ns,monotonic_after_ns=ns,
                utc=(fixtures.BASE+timedelta(seconds=seconds)).isoformat())


def receipt(pid,start,end,collector=52,parent=50):
    fixture=processes.ClockProcessValidationTests(); fixture.setUp()
    value=deepcopy(fixture.result['jobs'][0]['receipt'])
    value.update(pid=pid,started_utc=stamp(start)['utc'],finished_utc=stamp(end)['utc'],
                 elapsed_seconds_including_cleanup=end-start,spawn_returned_monotonic_seconds=1000+start+.001,
                 lifetime=dict(pid=collector,parent_pid=parent,start=stamp(start),end=stamp(end)))
    return value


class ClockTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu=fixtures.cpu_fixture(); cls.gpu=fixtures.gpu_fixture()

    def setUp(self):
        self.jobs=[dict(ordinal=i+1,runtime=r,repeat=i//3+1,receipt=receipt(100+i,10+20*i,11+20*i))
                   for i,r in enumerate(ORDER)]
        self.sequence=dict(schema_version=1,status='complete',stop_reason=None,external_cancelled=False,
                           diagnostic_only=True,jobs=self.jobs)
        self.outer=receipt(50,2,500,49,48)
        self.cpu_path='/telemetry/cpu.jsonl'; self.cpu_argv=['/python','-S','/sampler.py',self.cpu_path]
        cpu_receipt=receipt(62,-1,901,61,60); cpu_receipt['argv']=self.cpu_argv
        cpu_receipt['report_capture']=dict(path=self.cpu_path,size_bytes=len(self.cpu),
            sha256=hashlib.sha256(self.cpu).hexdigest(),device=1,inode=1,
            post_exit_hash_verified=True,producer_identity_verified=False)
        self.telemetry=dict(schema_version=1,status='complete',ready=True,errors=[],interrupt_signal=None,
            telemetry_contents_qualified=False,sequence_coverage_verified=False,launch_provenance_verified=False,
            cpu_receipt=cpu_receipt,gpu_sha256=hashlib.sha256(self.gpu).hexdigest(),gpu_pid=63,
            lifetime=dict(pid=61,parent_pid=60,start=stamp(-2),end=stamp(902)))
        self.ready=dict(schema_version=1,kind='telemetry_ready',worker_pid=61,gpu_pid=63,observer_affinity=[0],
            cpu_metadata=json.loads(self.cpu.splitlines()[0]),
            first_gpu_row=list(csv.reader(io.StringIO(self.gpu.decode())))[1],
            local_time_with_offset=stamp(1)['utc'],record_clock=stamp(1))

    def validate(self,cpu=None,gpu=None,**changes):
        args=dict(expected_cpu_source=fixtures.SOURCE,gpu_uuid=fixtures.UUID,timezone_name='UTC',
                  cpu_path=self.cpu_path,cpu_argv=self.cpu_argv)
        args.update(changes)
        return validate(self.cpu if cpu is None else cpu,self.gpu if gpu is None else gpu,
                        self.sequence,self.outer,self.telemetry,self.ready,**args)

    def bind_raw(self,cpu=None,gpu=None):
        if cpu is not None:
            self.telemetry['cpu_receipt']['report_capture'].update(size_bytes=len(cpu),sha256=hashlib.sha256(cpu).hexdigest())
            self.ready['cpu_metadata']=json.loads(cpu.splitlines()[0])
        if gpu is not None:
            self.telemetry['gpu_sha256']=hashlib.sha256(gpu).hexdigest()
            self.ready['first_gpu_row']=list(csv.reader(io.StringIO(gpu.decode())))[1]

    def test_full_serializer_901_samples_eighteen_jobs_and_readiness_no_processes(self):
        before=deepcopy((self.sequence,self.outer,self.telemetry,self.ready))
        with patch('subprocess.Popen',side_effect=AssertionError('no processes')):
            value=self.validate()
        self.assertTrue(value['telemetry_contents_validated']); self.assertTrue(value['readiness_contents_bound'])
        self.assertEqual(value['cpu_completion']['samples'],901); self.assertEqual(len(value['jobs']),18)
        self.assertEqual(value['restorer']['pid'],50); self.assertEqual(value['cpu_sha256'],hashlib.sha256(self.cpu).hexdigest())
        self.assertEqual(len(value['missing_cpu_optional']['pressure']),2703)
        for key in ('launch_provenance_verified','continuous_gpu_observation_verified','clock_enforcement_verified',
                    'hardware_policy_restoration_verified','performance_acceptance_evaluated','benchmark_outputs_qualified',
                    'postflight_call_order_enforced'): self.assertFalse(value[key])
        self.assertEqual(before,(self.sequence,self.outer,self.telemetry,self.ready))

    def test_raw_receipts_readiness_and_external_declaration_mismatches(self):
        for mode in ('cpu_hash','cpu_size','cpu_path','cpu_argv','producer','gpu_hash','metadata','gpu_first',
                     'ready_pid','ready_gpu','affinity','zone','future_local','past_local','future_gpu','ready_clock','scope','source','uuid'):
            self.setUp(); kwargs={}; gpu=None
            if mode=='cpu_hash': self.telemetry['cpu_receipt']['report_capture']['sha256']='0'*64
            if mode=='cpu_size': self.telemetry['cpu_receipt']['report_capture']['size_bytes']+=1
            if mode=='cpu_path': kwargs['cpu_path']='/other.jsonl'
            if mode=='cpu_argv': kwargs['cpu_argv']=['/other']
            if mode=='producer': self.telemetry['cpu_receipt']['report_capture']['producer_identity_verified']=True
            if mode=='gpu_hash': self.telemetry['gpu_sha256']='0'*64
            if mode=='metadata': self.ready['cpu_metadata']['tick_rate_hz']+=1
            if mode=='gpu_first': self.ready['first_gpu_row'][2]='11 %'
            if mode=='ready_pid': self.ready['worker_pid']+=1
            if mode=='ready_gpu': self.ready['gpu_pid']+=1
            if mode=='affinity': self.ready['observer_affinity']=[False]
            if mode=='zone': self.ready['local_time_with_offset']='2026-11-11T13:00:01+01:00'
            if mode=='future_local': self.ready['local_time_with_offset']=stamp(2)['utc']
            if mode=='past_local': self.ready['local_time_with_offset']=stamp(-1)['utc']
            if mode=='future_gpu':
                rows=list(csv.reader(io.StringIO(self.gpu.decode())))
                for index,row in enumerate(rows[1:]):
                    row[0]=(fixtures.BASE+timedelta(seconds=index+1.5)).strftime('%Y/%m/%d %H:%M:%S.%f')
                gpu=fixtures.gpu_bytes(rows); self.bind_raw(gpu=gpu)
            if mode=='ready_clock': self.ready['record_clock']=stamp(3)
            if mode=='scope': self.telemetry['telemetry_contents_qualified']=True
            if mode=='source': kwargs['expected_cpu_source']='0'*64
            if mode=='uuid': kwargs['gpu_uuid']='GPU-00000000-0000-0000-0000-000000000000'
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(gpu=gpu,**kwargs)

    def test_short_truncated_and_rehashed_bad_cpu_data_remain_invalid(self):
        for mode in ('short','truncated','activity','topology','completion','clock','counter'):
            self.setUp(); rows=[json.loads(line) for line in self.cpu.splitlines()]
            if mode=='short': rows.pop(10)
            if mode=='activity': rows[2]['activity_since_previous']['cpu']['total_ticks']+=1
            if mode=='topology': rows[0]['smt_siblings']=[2,4]
            if mode=='completion': rows[-1]['actual_elapsed_seconds']=899
            if mode=='clock': rows[2]['utc']=stamp(3)['utc']
            if mode=='counter': rows[2]['system_counters']['ctxt']=-1
            raw=fixtures.cpu_bytes(rows)
            if mode=='truncated': raw=raw[:-1]
            self.bind_raw(cpu=raw)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(cpu=raw)

    def test_rehashed_bad_gpu_units_uuid_time_and_coverage(self):
        for mode in ('units','uuid','duplicate','nonfinite','truncated','start','end'):
            self.setUp(); rows=list(csv.reader(io.StringIO(self.gpu.decode())))
            if mode=='units': rows[2][6]='1710 kHz'
            if mode=='uuid': rows[2][1]='GPU-00000000-0000-0000-0000-000000000000'
            if mode=='duplicate': rows[2]=deepcopy(rows[1])
            if mode=='nonfinite': rows[2][5]='NaN W'
            if mode=='start': rows=[rows[0],*rows[4:]]  # Jobs fit; the outer restorer starts before this prefix.
            if mode=='end': rows=rows[:490]  # Jobs fit; the restorer still exceeds the last sample.
            raw=fixtures.gpu_bytes(rows)
            if mode=='truncated': raw=raw[:-1]
            self.bind_raw(gpu=raw)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(gpu=raw)

    def test_receipt_order_identity_completeness_and_enclosure(self):
        for mode in ('partial','runtime','repeat','pid','outer_pid','exit','overlap','outside','late_outer','schema'):
            self.setUp()
            if mode=='partial': self.jobs.pop()
            if mode=='runtime': self.jobs[0]['runtime']='candidate'
            if mode=='repeat': self.jobs[0]['repeat']=True
            if mode=='pid': self.jobs[1]['receipt']['pid']=100
            if mode=='outer_pid': self.jobs[0]['receipt']['pid']=50
            if mode=='exit': self.jobs[0]['receipt']['returncode']=1
            if mode=='overlap': self.jobs[1]['receipt']=receipt(101,10.5,12)
            if mode=='outside': self.jobs[-1]['receipt']=receipt(117,501,502)
            if mode=='late_outer': self.outer=receipt(50,2,901,49,48)
            if mode=='schema': self.sequence['schema_version']=True
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate()

    def test_sparse_missing_gpu_values_remain_visible_not_enforcement(self):
        rows=list(csv.reader(io.StringIO(self.gpu.decode())))
        rows[1][6]='[Not Supported]'; rows[501][5]='N/A'
        rows=[rows[0],rows[1],rows[501],rows[901]]
        raw=fixtures.gpu_bytes(rows); self.bind_raw(gpu=raw)
        value=self.validate(gpu=raw)
        self.assertEqual(value['gpu_sample_gaps_seconds'],[500.0,400.0])
        self.assertFalse(value['gpu_samples'][0]['sm_mhz']['available'])
        self.assertFalse(value['gpu_samples'][1]['power_w']['available'])
        self.assertTrue(all(not row['gpu_sample_indices_inside'] for row in value['jobs']))
        self.assertFalse(value['clock_enforcement_verified'])
        self.assertFalse(value['continuous_gpu_observation_verified'])


if __name__ == '__main__': unittest.main()
