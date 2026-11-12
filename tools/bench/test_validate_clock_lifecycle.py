"""Synthetic tampering plus real sibling-supervisor fake-policy transport."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import test_validate_clock_restoration as transaction
import test_clock_diagnostic_coordinator as live
from test_verify_clock_processes import stamp
from clock_process_observation import ORDER
from validate_readonly_lifecycle import digest
from validate_clock_lifecycle import validate


def receipt(template,pid,start,end,collector,parent):
    value=deepcopy(template)
    start_ns,end_ns=int(start*10**9),int(end*10**9)
    value.update(pid=pid,lifetime=dict(pid=collector,parent_pid=parent,start=stamp(start_ns),end=stamp(end_ns)),
        started_utc=stamp(start_ns)['utc'],finished_utc=stamp(end_ns)['utc'],
        elapsed_seconds_including_cleanup=(end_ns-start_ns)/1e9,spawn_returned_monotonic_seconds=start+.001)
    return value


def cpu_row(index,seconds):
    value=stamp(seconds*10**9)
    return dict(kind='sample',index=index,monotonic_start_ns=value['monotonic_before_ns'],
                monotonic_end_ns=value['monotonic_after_ns'],utc=value['utc'])


class ClockLifecycleTests(unittest.TestCase):
    def setUp(self):
        fixture=transaction.RestorationValidationTests(); fixture.setUp()
        self.commands=fixture.commands
        self.restoration,self.obligation,self.sequence=deepcopy((fixture.restoration,fixture.obligation,fixture.sequence))
        template=self.restoration['set']
        self.outer=receipt(template,50,99,121,49,48)
        self.outer.update(timeout_seconds=660,termination_grace_seconds=60)
        worker=receipt(template,61,93,994,60,49)
        self.supervisor=dict(schema_version=1,status='complete',errors=[],parent_lost=False,orphan_observed=False,
            subreaper_restored=True,interrupt_signal=None,launch_provenance_verified=False,
            benchmark_outputs_qualified=False,worker=worker,lifetime=transaction.life(92,995,60,49),
            cleanup=dict(complete=True,signalled=[],reaped=[],remaining=[],errors=[]))
        self.telemetry=dict(schema_version=1,status='complete',ready=True,errors=[],interrupt_signal=None,
            telemetry_contents_qualified=False,sequence_coverage_verified=False,launch_provenance_verified=False,
            cpu_receipt=receipt(template,62,95,980,61,60),gpu_pid=63,
            gpu_start_clock=stamp(94*10**9),gpu_stop_clock=stamp(981*10**9),
            gpu_shutdown=dict(verified=True,returncode=-15,kill_escalation=False),lifetime=transaction.life(94,990,61,60))
        self.ready=dict(schema_version=1,kind='telemetry_ready',worker_pid=61,gpu_pid=63,observer_affinity=[0],
                        record_clock=stamp(97*10**9))
        self.collection=dict(schema_version=1,status='complete',errors=[],interrupt_signal=None,
            restoration_supervisor_status='complete',telemetry_status='complete',observer_pid=60,
            observer_shutdown=dict(verified=True,returncode=0,kill_escalation=False),
            launch_provenance_verified=False,telemetry_contents_qualified=False,sequence_coverage_verified=False,
            benchmark_outputs_qualified=False,ready_canonical_sha256=digest(self.ready),
            ready_observed_clock=stamp(98*10**9),lifetime=transaction.life(90,1000,49,48))
        self.samples=[cpu_row(0,96),cpu_row(1,900)]
        self.sequence['jobs']=[]
        for i,runtime in enumerate(ORDER):
            start=105.5+i*.5; end=start+.1
            value=receipt(template,100+i,start,end,52,50); value['process_observer_enabled']=True
            self.sequence['jobs'].append(dict(ordinal=i+1,runtime=runtime,repeat=i//3+1,receipt=value))

    def records(self):
        return [self.collection,self.outer,self.restoration,self.obligation,self.sequence,
                self.supervisor,self.telemetry,self.ready,self.samples]

    def validate(self,records=None):
        return validate(*(self.records() if records is None else records),commands=self.commands,cwd='/checkout',
                        work_deadline=691,external_cancel_check_enabled=False)

    def test_complete_scope_no_io_and_independent_clock_domain(self):
        before=deepcopy(self.records())
        with patch('builtins.open',side_effect=AssertionError('no file IO')), \
             patch.object(Path,'open',side_effect=AssertionError('no file IO')), \
             patch('subprocess.Popen',side_effect=AssertionError('no processes')):
            value=self.validate()
        self.assertTrue(value['sibling_lifecycle_consistency_verified'])
        self.assertTrue(value['restoration_binding']['restoration_record_consistency_verified'])
        self.assertEqual(len(value['jobs']),18)
        for key in ('launch_provenance_verified','telemetry_contents_qualified','hardware_policy_restoration_verified',
                    'benchmark_outputs_qualified','postflight_call_order_enforced'): self.assertFalse(value[key])
        self.assertEqual(before,self.records())

    def test_parent_chains_and_duplicate_pids(self):
        for mode in ('restorer','observer','telemetry','cpu_collector','worker_receipt','ready_worker','gpu','job','collector'):
            rows=deepcopy(self.records()); c,o,r,b,s,sup,t,ready,samples=rows
            if mode=='restorer': r['lifetime']['parent_pid']=60; b['supervisor_parent_pid']=60
            if mode=='observer': sup['lifetime']['parent_pid']=50
            if mode=='telemetry': t['lifetime']['parent_pid']=50
            if mode=='cpu_collector': t['cpu_receipt']['lifetime']['pid']=60
            if mode=='worker_receipt': sup['worker']['pid']=62
            if mode=='ready_worker': ready['worker_pid']=60
            if mode=='gpu': t['gpu_pid']=ready['gpu_pid']=62
            if mode=='job': s['jobs'][0]['receipt']['pid']=63
            if mode=='collector': o['lifetime']['pid']=60
            c['ready_canonical_sha256']=digest(ready)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(rows)

    def test_status_cleanup_limits_and_false_scope(self):
        for mode in ('coord','legacy','observer','cleanup','reap','gpu','telemetry','claim','limit','grace','bytes','timeout',
                     'worker_exit','worker_term','cpu_term','cpu_capture'):
            rows=deepcopy(self.records()); c,o,r,b,s,sup,t,ready,samples=rows
            if mode=='coord': c['restoration_supervisor_status']='failed'
            if mode=='legacy': c['sequence_status']='complete'
            if mode=='observer': sup['parent_lost']=True
            if mode=='cleanup': sup['cleanup']['reaped']=[{'pid':999}]
            if mode=='reap': c['observer_shutdown']['kill_escalation']=True
            if mode=='gpu': t['gpu_shutdown']['returncode']=-9
            if mode=='telemetry': t['errors']=['missing sample']
            if mode=='claim': c['sequence_coverage_verified']=True
            if mode=='limit': o['output_limit_bytes']=2*1024**2
            if mode=='grace': o['termination_grace_seconds']=10
            if mode=='bytes': o['observed_bytes']=1
            if mode=='timeout': o['timeout_seconds']=700
            if mode=='worker_exit': sup['worker']['returncode']=1
            if mode=='worker_term': sup['worker']['termination_signal']='sent'
            if mode=='cpu_term': t['cpu_receipt']['termination_signal']='sent'
            if mode=='cpu_capture': t['cpu_receipt']['observed_bytes']=1
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(rows)

    def test_readiness_lifetimes_cpu_brackets_and_order(self):
        for mode in ('hash','late_ready','affinity','gpu_start','gpu_end','cpu_first','cpu_last','cpu_order',
                     'cpu_utc','outside','partial','ordinal','repeat','unobserved','overlap'):
            rows=deepcopy(self.records()); c,o,r,b,s,sup,t,ready,samples=rows
            if mode=='hash': c['ready_canonical_sha256']='0'*64
            if mode=='late_ready': c['ready_observed_clock']=stamp(100*10**9)
            if mode=='affinity': ready['observer_affinity']=[2]; c['ready_canonical_sha256']=digest(ready)
            if mode=='gpu_start': t['gpu_start_clock']=stamp(96*10**9)
            if mode=='gpu_end': t['gpu_stop_clock']=stamp(979*10**9)
            if mode=='cpu_first': samples[0]=cpu_row(0,100)
            if mode=='cpu_last': samples[-1]=cpu_row(1,120)
            if mode=='cpu_order': samples[-1]['index']=0
            if mode=='cpu_utc': samples[-1]['utc']='2000-01-01T00:00:00+00:00'
            if mode=='outside': sup['lifetime']['end']=stamp(1001*10**9)
            if mode=='partial': s['jobs'].pop()
            if mode=='ordinal': s['jobs'][0]['ordinal']=True
            if mode=='repeat': s['jobs'][0]['repeat']=2
            if mode=='unobserved': s['jobs'][0]['receipt']['process_observer_enabled']=False
            if mode=='overlap':
                a=s['jobs'][0]['receipt']; second=s['jobs'][1]['receipt']
                for key in ('lifetime','started_utc','finished_utc','spawn_returned_monotonic_seconds'):
                    second[key]=deepcopy(a[key])
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(rows)

    def test_real_sibling_supervisors_and_default_budget_fake_policy(self):
        if getattr(live.ClockCoordinatorTests,'__unittest_skip__',False): self.skipTest('Linux CPU0/2 and taskset required')
        fixture=live.ClockCoordinatorTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root,_=fixture.run_case('lifecycle_validation')
        read=lambda name: json.loads((root/'collection'/name).read_bytes())
        rows=[read(p) for p in ('collection.json','restorer.receipt.json','restoration/restoration.json',
              'restoration/obligation.json','sequence/sequence-result.json','observer-supervision/supervisor.json',
              'telemetry/telemetry.json','telemetry/ready.json')]
        samples=[json.loads(line) for line in (root/'collection/telemetry/cpu.jsonl').read_text().splitlines()]
        rows.append([row for row in samples if row['kind']=='sample'])
        python=str(Path(sys.executable).resolve()); policy=[python,'-S',str(root/'policy.py'),str(root)]
        commands={'set':policy+['set'],'reset':policy+['reset'],
                  'worker':[python,'-S',str(root/'worker.py'),str(live.HERE),str(root),'success',
                            str(Path(shutil.which('taskset')).resolve())]}
        received=json.loads((root/'received-deadline.json').read_bytes())
        value=validate(*rows,commands=commands,cwd=str(live.HERE),work_deadline=received['deadline'],
                       external_cancel_check_enabled=False)
        self.assertEqual(len(value['jobs']),18)
        pids=[rows[0]['lifetime']['pid'],rows[0]['observer_pid'],rows[1]['pid'],rows[6]['lifetime']['pid'],
              rows[6]['cpu_receipt']['pid'],rows[6]['gpu_pid']]
        pids += [row['pid'] for row in value['jobs']]
        self.assertTrue(all(not Path('/proc/'+str(pid)).exists() for pid in pids))
        self.assertFalse(value['telemetry_contents_qualified'])


if __name__ == '__main__': unittest.main()
