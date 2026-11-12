"""Offline tampering checks and real default-budget fake-policy CPU transport."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_verify_clock_processes as process_fixture
from test_verify_clock_processes import stamp
import test_clock_worker_handoff as live
from validate_clock_restoration import validate


def life(start,end,pid=50,parent=49):
    return dict(pid=pid,parent_pid=parent,start=stamp(start*10**9),end=stamp(end*10**9))


class RestorationValidationTests(unittest.TestCase):
    def setUp(self):
        fixture=process_fixture.ClockProcessValidationTests(); fixture.setUp()
        template=fixture.result['jobs'][0]['receipt']
        self.commands={k:['/python','-S',k] for k in ('set','worker','reset')}
        receipts={}
        for label,pid,start,end in (('set',51,102,103),('worker',52,104,117),('reset',53,118,119)):
            receipt=deepcopy(template)
            receipt.update(pid=pid,argv=self.commands[label],cwd='/checkout',process_observer_enabled=False,
                report_capture=None,report_limit_bytes=None,timeout_seconds=600 if label=='worker' else 10,
                lifetime=life(start,end),started_utc=stamp(start*10**9)['utc'],finished_utc=stamp(end*10**9)['utc'],
                elapsed_seconds_including_cleanup=end-start,spawn_returned_monotonic_seconds=start+.001)
            receipts[label]=receipt
        self.obligation=dict(schema_version=2,armed_before_set=True,supervisor_pid=50,supervisor_parent_pid=49,
            lock_seconds=600,command_seconds=10,cleanup_reserve_seconds=10,maximum_reset_attempts=2,
            work_deadline_monotonic_seconds=691,reset_deadline_monotonic_seconds=701,clock=stamp(101*10**9),
            **{k+'_argv':v for k,v in self.commands.items()})
        self.restoration=dict(schema_version=2,status='complete',armed=True,errors=[],parent_lost=False,
            orphan_observed=False,external_cancelled=False,external_cancel_check_enabled=False,
            interrupt_signal=None,subreaper_restored=True,reset_command_succeeded=True,
            reset_request_within_deadline=True,set=receipts['set'],worker=receipts['worker'],
            reset_attempts=[receipts['reset']],reset_started_clock=stamp(117*10**9),
            work_deadline_monotonic_seconds=691,reset_deadline_monotonic_seconds=701,cleanup_reserve_seconds=10,
            lifetime=life(100,120),cleanups=[dict(after=k,result=dict(complete=True,signalled=[],reaped=[],remaining=[],errors=[]))
                                          for k in ('set','before_reset','reset-1')])
        for k in ('hardware_policy_restoration_verified','clock_enforcement_verified','launch_provenance_verified',
                  'benchmark_outputs_qualified'): self.restoration[k]=False
        self.sequence=dict(schema_version=1,status='complete',stop_reason=None,external_cancelled=False,
                           work_deadline_monotonic_seconds=691,lifetime=life(105,116,52,50))

    def validate(self, restoration=None, obligation=None, sequence=None, **kw):
        args=dict(commands=self.commands,cwd='/checkout',work_deadline=691,external_cancel_check_enabled=False)
        args.update(kw)
        return validate(self.restoration if restoration is None else restoration,
                        self.obligation if obligation is None else obligation,
                        self.sequence if sequence is None else sequence,**args)

    def test_valid_scope_no_io_and_no_mutation(self):
        before=deepcopy((self.restoration,self.obligation,self.sequence,self.commands))
        with patch('builtins.open',side_effect=AssertionError('no file reads')), \
             patch.object(Path,'open',side_effect=AssertionError('no file reads')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no filesystem')), \
             patch('subprocess.Popen',side_effect=AssertionError('no processes')):
            result=self.validate()
        self.assertTrue(result['restoration_record_consistency_verified'])
        self.assertEqual(result['worker_pid'],52)
        self.assertEqual(set(result['records_canonical_sha256']),{'restoration','obligation','sequence'})
        for k in ('hardware_policy_restoration_verified','clock_enforcement_verified','outer_supervision_verified',
                  'observer_coverage_verified','launch_provenance_verified','benchmark_outputs_qualified','postflight_call_order_enforced'):
            self.assertFalse(result[k])
        self.assertEqual(before,(self.restoration,self.obligation,self.sequence,self.commands))

    def test_status_cleanup_scope_and_original_failure_cannot_be_erased(self):
        modes=('status','armed','error','parent','orphan','cancel','hook','interrupt','role','reset','deadline_flag',
               'claim','cleanup_missing','cleanup_order','cleanup_signal','cleanup_failed','retry','schema')
        for mode in modes:
            value=deepcopy(self.restoration)
            if mode=='status': value['status']='failed'
            if mode=='armed': value['armed']=False
            if mode=='error': value['errors']=['reset_attempt_failed:1']
            if mode=='parent': value['parent_lost']=True
            if mode=='orphan': value['orphan_observed']=True
            if mode=='cancel': value['external_cancelled']=True
            if mode=='hook': value['external_cancel_check_enabled']=True
            if mode=='interrupt': value['interrupt_signal']=15
            if mode=='role': value['subreaper_restored']=False
            if mode=='reset': value['reset_command_succeeded']=False
            if mode=='deadline_flag': value['reset_request_within_deadline']=False
            if mode=='claim': value['hardware_policy_restoration_verified']=True
            if mode=='cleanup_missing': value['cleanups'].pop()
            if mode=='cleanup_order': value['cleanups'].reverse()
            if mode=='cleanup_signal': value['cleanups'][0]['result']['signalled']=[99]
            if mode=='cleanup_failed': value['cleanups'][1]['result']['complete']=False
            if mode=='retry': value['reset_attempts'].append(deepcopy(value['reset_attempts'][0]))
            if mode=='schema': value['schema_version']=True
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(restoration=value)

    def test_deadline_writer_clock_and_command_tampering(self):
        for mode in ('budget','reserve','attempts','arm','writer','parent','command','work','reset','clock','utc','nonfinite'):
            value=deepcopy(self.obligation)
            if mode=='budget': value['lock_seconds']=590
            if mode=='reserve': value['cleanup_reserve_seconds']=True
            if mode=='attempts': value['maximum_reset_attempts']=1
            if mode=='arm': value['armed_before_set']=False
            if mode=='writer': value['supervisor_pid']=52
            if mode=='parent': value['supervisor_parent_pid']=1
            if mode=='command': value['reset_argv']=['/unapproved']
            if mode=='work': value['work_deadline_monotonic_seconds']+=1
            if mode=='reset': value['reset_deadline_monotonic_seconds']+=1
            if mode=='clock': value['clock']=stamp(103*10**9)
            if mode=='utc': value['clock']['utc']='2000-01-01T00:00:00+00:00'
            if mode=='nonfinite': value['work_deadline_monotonic_seconds']=float('nan')
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(obligation=value)
        with self.assertRaises(ValueError): self.validate(work_deadline=692)

    def test_receipt_commands_capture_clocks_order_and_deadline_flags_not_trusted(self):
        for mode in ('exit','pid','command','cwd','timeout','limit','grace','observer','capture','hash','spawn','parent','reset_clock','late'):
            value=deepcopy(self.restoration); receipt=value['reset_attempts'][0]
            if mode=='exit': receipt['returncode']=1
            if mode=='pid': receipt['pid']=52
            if mode=='command': receipt['argv']=['/wrong']
            if mode=='cwd': receipt['cwd']='/other'
            if mode=='timeout': receipt['timeout_seconds']=11
            if mode=='limit': receipt['output_limit_bytes']=True
            if mode=='grace': receipt['termination_grace_seconds']=1
            if mode=='observer': receipt['process_observer_enabled']=True
            if mode=='capture': receipt['observed_bytes']=1
            if mode=='hash': receipt['capture_sha256']='bad'
            if mode=='spawn': receipt['spawn_returned_monotonic_seconds']=float('inf')
            if mode=='parent': receipt['lifetime']['parent_pid']=48
            if mode=='reset_clock': value['reset_started_clock']=stamp(110*10**9)
            if mode=='late':
                # Keep internal clocks consistent and the green deadline flag;
                # rejection must come from the original absolute reset deadline.
                receipt.update(lifetime=life(702,703),started_utc=stamp(702*10**9)['utc'],
                               finished_utc=stamp(703*10**9)['utc'],spawn_returned_monotonic_seconds=702.001)
                value['reset_started_clock']=stamp(702*10**9); value['lifetime']=life(100,704)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(restoration=value)

    def test_sequence_identity_enclosure_and_original_deadline(self):
        for mode in ('pid','parent','start','end','deadline','status','cancel'):
            value=deepcopy(self.sequence)
            if mode=='pid': value['lifetime']['pid']=51
            if mode=='parent': value['lifetime']['parent_pid']=49
            if mode=='start': value['lifetime']['start']=stamp(103*10**9)
            if mode=='end': value['lifetime']['end']=stamp(118*10**9)
            if mode=='deadline': value['work_deadline_monotonic_seconds']=692
            if mode=='status': value['status']='failed'
            if mode=='cancel': value['external_cancelled']=True
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.validate(sequence=value)

    def test_real_default_budget_handoff_with_eighteen_cpu_jobs(self):
        if getattr(live.ClockHandoffTests,'__unittest_skip__',False): self.skipTest('Linux CPU0/2 and taskset required')
        fixture=live.ClockHandoffTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root,_=fixture.run_case('success')
        read=lambda name: json.loads((root/name).read_bytes())
        record=read('restoration/restoration.json'); obligation=read('restoration/obligation.json')
        sequence=read('sequence/sequence-result.json'); handoff=read('handoff.json')
        result=validate(record,obligation,sequence,commands=handoff['commands'],cwd=str(live.HERE),
                        work_deadline=read('received-deadline.json')['deadline'],external_cancel_check_enabled=False)
        self.assertTrue(result['restoration_record_consistency_verified'])
        self.assertEqual(len(sequence['jobs']),18)
        pids=[result[k] for k in ('supervisor_pid','worker_pid','set_pid','reset_pid')]
        pids += [j['receipt']['pid'] for j in sequence['jobs']]
        self.assertTrue(all(not Path('/proc/'+str(pid)).exists() for pid in pids))
        self.assertFalse(result['hardware_policy_restoration_verified'])


if __name__ == '__main__': unittest.main()
