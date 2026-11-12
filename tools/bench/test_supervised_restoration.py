"""Real owned subprocesses, fake set/reset state only; NEVER GPU commands."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ENV = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'LANG': 'C', 'LC_ALL': 'C'}
FAKE = '''import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]); stage=sys.argv[2]; mode=sys.argv[3]
with (root/'events').open('a') as f:f.write(stage+'\\n')
if stage=='set':
    assert json.loads((root/'transaction/obligation.json').read_bytes())['armed_before_set'] is True
    (root/'lock').write_text('fake policy')
    if mode=='observer_set':
        (root/'observer_failed').touch();time.sleep(30)
    if mode=='set_fail':raise SystemExit(2)
    if mode=='set_timeout':time.sleep(30)
    if mode=='set_output':sys.stdout.write('x'*(1024**2+65536));sys.stdout.flush();time.sleep(30)
elif stage=='worker':
    assert (root/'lock').exists()
    (root/'worker.pid').write_text(str(os.getpid()))
    if mode in ('observer_worker','observer_error'):
        (root/'observer_failed').touch();time.sleep(30)
    if mode=='worker_fail':raise SystemExit(7)
    if mode=='worker_output':sys.stdout.write('x'*(1024**2+65536));sys.stdout.flush();time.sleep(30)
    if mode=='crash':
        code='from pathlib import Path; import os,time; Path('+repr(str(root/'escaped.pid'))+').write_text(str(os.getpid())); time.sleep(30)'
        subprocess.Popen([sys.executable,'-S','-c',code],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        end=time.monotonic()+3
        while not (root/'escaped.pid').exists():
            assert time.monotonic()<end
            time.sleep(.01)
        os.kill(os.getpid(),signal.SIGKILL)
    if mode=='interrupt':os.kill(os.getppid(),signal.SIGTERM)
    if mode in ('interrupt','timeout','parent_loss'):time.sleep(30)
elif stage=='reset':
    # Reset must follow actual owned worker/escaped-child reaping, not a flag.
    for name in ('worker.pid','escaped.pid'):
        path=root/name
        if path.exists():assert not Path('/proc/'+path.read_text()).exists()
    count=(root/'events').read_text().splitlines().count('reset')
    if mode=='reset_fail' or (mode=='reset_retry' and count==1):raise SystemExit(9)
    if mode=='reset_timeout':time.sleep(30)
    if mode=='reset_interrupt' and count==1:
        os.kill(os.getppid(),signal.SIGTERM);time.sleep(30)
    (root/'lock').unlink(missing_ok=True)
else:raise AssertionError(stage)
'''


def code(root, mode, parent, *, lock_seconds=3, command_seconds=.5,
         cleanup_reserve_seconds=.1, inject=None):
    fake = root / 'fake.py'
    fake.write_text(FAKE)
    commands = [[str(Path(sys.executable).resolve()), '-S', str(fake), str(root), stage, mode]
                for stage in ('set', 'worker', 'reset')]
    prefix = 'import json,os,signal\nfrom unittest.mock import patch\nimport supervised_restoration as s\n'
    call = (f's.supervise_restoration(*{commands!r},cwd={str(root)!r},env={ENV!r},'
            f'output_dir={str(root / "transaction")!r},parent_pid={parent},'
            f'lock_seconds={lock_seconds!r},command_seconds={command_seconds!r},'
            f'cleanup_reserve_seconds={cleanup_reserve_seconds!r})')
    if inject in ('external_before', 'external_watch', 'external_error', 'external_invalid', 'external_late'):
        call = call[:-1] + ',external_cancel_check=check)'
        prefix += f'''root=s.Path({str(root)!r})
def check():
    events=(root/'events').read_text().splitlines() if (root/'events').exists() else []
    assert 'reset' not in events,'cancellation must not run during reset'
    if {inject!r}=='external_before':return True
    if {inject!r}=='external_invalid':return None
    failed=(root/'observer_failed').exists()
    if failed and {inject!r}=='external_error':raise OSError('synthetic observer failure')
    if failed and {inject!r}=='external_watch':(root/'observer_failed').unlink()
    return failed
'''
        if inject == 'external_late':
            prefix += '''actual_capture=s.run_bounded
def capture(argv,**kwargs):
    receipt=actual_capture(argv,**kwargs)
    if kwargs['capture_path'].name=='worker.txt':(root/'observer_failed').touch()
    return receipt
s.run_bounded=capture
'''
    if inject == 'capture':
        prefix += '''actual=s.run_bounded
def capture(argv,**kwargs):
    if kwargs['capture_path'].name=='set.txt':raise OSError('synthetic capture/serialization error')
    return actual(argv,**kwargs)
s.run_bounded=capture
'''
    if inject in ('set_after', 'worker_after', 'reset_after'):
        label = {'set_after': 'set.txt', 'worker_after': 'worker.txt', 'reset_after': 'reset-1.txt'}[inject]
        prefix += f'''actual=s.run_bounded
def capture(argv,**kwargs):
    receipt=actual(argv,**kwargs)
    if kwargs['capture_path'].name=={label!r}:raise OSError('synthetic exception after command reaping')
    return receipt
s.run_bounded=capture
'''
    if inject == 'reset_clock':
        prefix += '''actual_capture=s.run_bounded
actual_stamp=s.diagnostic_clock.stamp
fail_next=False
def capture(argv,**kwargs):
    global fail_next
    receipt=actual_capture(argv,**kwargs)
    if kwargs['capture_path'].name=='worker.txt':fail_next=True
    return receipt
def stamp():
    global fail_next
    if fail_next:
        fail_next=False
        raise OSError('synthetic reset timestamp failure')
    return actual_stamp()
s.run_bounded=capture
s.diagnostic_clock.stamp=stamp
'''
    if inject in ('late_reset', 'missing_spawn'):
        prefix += f'''actual_capture=s.run_bounded
def capture(argv,**kwargs):
    if kwargs['capture_path'].name=='reset-1.txt' and {inject!r}=='late_reset':
        deadline=json.loads((s.Path({str(root)!r})/'transaction/obligation.json').read_bytes())['reset_deadline_monotonic_seconds']
        s.time.sleep(max(0,deadline-s.time.monotonic())+.02)
    receipt=actual_capture(argv,**kwargs)
    if kwargs['capture_path'].name=='reset-1.txt' and {inject!r}=='missing_spawn':
        receipt.pop('spawn_returned_monotonic_seconds',None)
    return receipt
s.run_bounded=capture
'''
    if inject == 'warning':
        prefix += '''def warning(*args,**kwargs):
    raise BrokenPipeError('synthetic lost parent warning pipe')
s.print=warning
'''
    if inject == 'arm':
        prefix += '''actual_open=s.Path.open
def opening(path,*args,**kwargs):
    if path.name=='obligation.json':raise OSError('synthetic obligation write failure')
    return actual_open(path,*args,**kwargs)
s.Path.open=opening
'''
    if inject == 'cleanup':
        prefix += "s.cleanup_owned=lambda: {'complete':False,'remaining':[],'errors':['synthetic failure']}\n"
    if inject == 'setup':
        prefix += '''actual_signal=s.signal.signal
def setup(sig,handler):
    if sig==signal.SIGTERM:raise OSError('synthetic handler failure')
    return actual_signal(sig,handler)
s.signal.signal=setup
'''
    return prefix + 'result=' + call + '''
assert s.direct_children()==[],result
s.subreaper(0)
assert result==json.loads(open(''' + repr(str(root / 'transaction/restoration.json')) + ''').read()),result
print(json.dumps(result))
'''


@unittest.skipUnless(sys.platform == 'linux', 'Linux owned subreaper contract')
class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_case(self, mode, **kwargs):
        root = self.root / (mode + '-' + str(len(list(self.root.iterdir()))))
        root.mkdir()
        script = code(root, mode, os.getpid(), **kwargs)
        run = subprocess.run([sys.executable, '-S', '-c', script], cwd=HERE, env=ENV,
                             capture_output=True, text=True, check=True, timeout=15)
        result = json.loads(run.stdout)
        events = (root / 'events').read_text().splitlines() if (root / 'events').exists() else []
        self.assertFalse(result['hardware_policy_restoration_verified'])
        self.assertFalse(result['clock_enforcement_verified'])
        self.assertFalse(result['benchmark_outputs_qualified'])
        for receipt in [result['set'], result['worker'], *result['reset_attempts']]:
            if receipt and receipt['pid']:
                self.assertTrue(receipt['direct_child_reaped'])
                self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())
        return root, result, events, run.stderr

    def test_failed_warning_pipe_does_not_skip_role_cleanup_or_retained_failure(self):
        root, result, events, _ = self.run_case('reset_fail', inject='warning')
        self.assertEqual(events, ['set', 'worker', 'reset', 'reset'])
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['reset_command_succeeded'])
        self.assertTrue(result['subreaper_restored'])
        self.assertTrue((root / 'lock').exists())
        self.assertIn('warning_delivery_error:BrokenPipeError', result['errors'])

    def test_success_arms_before_set_and_resets_after_reaping(self):
        root, result, events, _ = self.run_case('success')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(events, ['set', 'worker', 'reset'])
        self.assertTrue(result['reset_command_succeeded'])
        self.assertTrue(result['subreaper_restored'])
        self.assertFalse((root / 'lock').exists())

    def test_cleanup_reserve_and_absolute_deadlines_are_armed_before_set(self):
        root, result, events, _ = self.run_case('success', cleanup_reserve_seconds=1)
        obligation = json.loads((root / 'transaction/obligation.json').read_bytes())
        self.assertEqual(events, ['set', 'worker', 'reset'])
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['reset_request_within_deadline'])
        self.assertEqual(obligation['cleanup_reserve_seconds'], 1)
        for key in ('reset_deadline_monotonic_seconds', 'work_deadline_monotonic_seconds'):
            self.assertEqual(result[key], obligation[key])
        self.assertEqual(result['reset_deadline_monotonic_seconds'] - result['work_deadline_monotonic_seconds'], 1)
        spawn = result['reset_attempts'][0]['spawn_returned_monotonic_seconds']
        self.assertLessEqual(spawn, result['reset_deadline_monotonic_seconds'])

    def test_work_deadline_leaves_time_for_actual_reset_request(self):
        root, result, events, _ = self.run_case('timeout', lock_seconds=1.5, cleanup_reserve_seconds=.8)
        self.assertEqual(events, ['set', 'worker', 'reset'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('sequence_deadline', result['worker']['issues'])
        self.assertTrue(result['reset_request_within_deadline'])
        self.assertFalse((root / 'lock').exists())

    def test_late_or_missing_spawn_evidence_cannot_use_earlier_reset_timestamp(self):
        for inject in ('late_reset', 'missing_spawn'):
            with self.subTest(inject=inject):
                root, result, events, _ = self.run_case('success', lock_seconds=1, inject=inject)
                self.assertEqual(events, ['set', 'worker', 'reset'])
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reset_command_succeeded'])
                self.assertFalse(result['reset_request_within_deadline'])
                self.assertIn('reset_request_deadline_unverified_or_missed', result['errors'])
                self.assertFalse((root / 'lock').exists())
                # The earlier coordinator timestamp alone would wrongly pass.
                early = result['reset_started_clock']['monotonic_after_ns'] / 1e9
                self.assertLess(early, result['reset_deadline_monotonic_seconds'])

    def test_external_cancel_or_invalid_status_before_arm_prevents_all_commands(self):
        for inject in ('external_before', 'external_invalid'):
            with self.subTest(inject=inject):
                root, result, events, _ = self.run_case('success', inject=inject)
                self.assertEqual(events, [])
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['armed'])
                self.assertTrue(result['external_cancel_check_enabled'])
                self.assertTrue(result['external_cancelled'])
                self.assertFalse((root / 'transaction/obligation.json').exists())
                if inject == 'external_invalid':
                    self.assertIn('external_cancel_error:ValueError', result['errors'])

    def test_external_observer_loss_stops_set_or_worker_but_not_reset(self):
        _, healthy, events, _ = self.run_case('success', inject='external_watch')
        self.assertEqual(healthy['status'], 'complete')
        self.assertTrue(healthy['external_cancel_check_enabled'])
        self.assertFalse(healthy['external_cancelled'])
        self.assertEqual(events, ['set', 'worker', 'reset'])
        for mode, stage in (('observer_set', 'set'), ('observer_worker', 'worker')):
            with self.subTest(mode=mode):
                root, result, events, _ = self.run_case(mode, inject='external_watch')
                self.assertEqual(events, ['set', 'reset'] if stage == 'set' else ['set', 'worker', 'reset'])
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['external_cancelled'])
                self.assertIn('external_cancelled', result[stage]['issues'])
                self.assertFalse((root / 'observer_failed').exists(), 'transient loss must stay latched')
                self.assertTrue(result['reset_command_succeeded'])
                self.assertFalse((root / 'lock').exists())

    def test_external_check_exception_or_loss_after_worker_receipt_still_restores(self):
        for mode, inject in (('observer_error', 'external_error'), ('success', 'external_late')):
            with self.subTest(inject=inject):
                root, result, events, _ = self.run_case(mode, inject=inject)
                self.assertEqual(events, ['set', 'worker', 'reset'])
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['external_cancelled'])
                self.assertTrue(result['reset_command_succeeded'])
                self.assertFalse((root / 'lock').exists())
                if inject == 'external_error':
                    self.assertIn('external_cancel_error:OSError', result['errors'])
                    self.assertIn('capture_error:OSError', result['worker']['issues'])
                else:
                    self.assertEqual(result['worker']['status'], 'complete')

    def test_partial_set_failure_skips_worker_but_restores(self):
        root, result, events, _ = self.run_case('set_fail')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(events, ['set', 'reset'])
        self.assertEqual(result['set']['returncode'], 2)
        self.assertIsNone(result['worker'])
        self.assertTrue(result['reset_command_succeeded'])
        self.assertFalse((root / 'lock').exists())

    def test_set_timeout_still_restores_without_worker(self):
        root, result, events, _ = self.run_case('set_timeout', command_seconds=.2)
        self.assertEqual(events, ['set', 'reset'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('child_timeout', result['set']['issues'])
        self.assertTrue(result['reset_command_succeeded'])
        self.assertFalse((root / 'lock').exists())

    def test_worker_failure_timeout_and_interrupt_all_restore(self):
        for mode in ('worker_fail', 'timeout', 'interrupt'):
            with self.subTest(mode=mode):
                root, result, events, _ = self.run_case(mode, lock_seconds=.4)
                self.assertEqual(events, ['set', 'worker', 'reset'])
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reset_command_succeeded'])
                self.assertFalse((root / 'lock').exists())
                if mode == 'timeout':
                    self.assertIn('sequence_deadline', result['worker']['issues'])
                if mode == 'interrupt':
                    self.assertEqual(result['interrupt_signal'], 15)

    def test_worker_crash_session_escape_reaped_before_reset_unrelated_survives(self):
        other = subprocess.Popen([sys.executable, '-S', '-c', 'import time;time.sleep(30)'], env=ENV)
        try:
            root, result, events, _ = self.run_case('crash')
            self.assertEqual(events, ['set', 'worker', 'reset'])
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['orphan_observed'])
            self.assertTrue(result['reset_command_succeeded'])
            self.assertIsNone(other.poll())
            self.assertFalse(Path('/proc/' + (root / 'escaped.pid').read_text()).exists())
        finally:
            other.kill()
            other.wait()

    def test_output_limit_on_set_or_worker_still_resets(self):
        for mode, stage in (('set_output', 'set'), ('worker_output', 'worker')):
            with self.subTest(mode=mode):
                root, result, events, _ = self.run_case(mode, command_seconds=2)
                self.assertEqual(result['status'], 'failed')
                self.assertIn('output_limit', result[stage]['issues'])
                self.assertTrue(result['reset_command_succeeded'])
                self.assertEqual(events, ['set', 'reset'] if stage == 'set' else ['set', 'worker', 'reset'])
                self.assertFalse((root / 'lock').exists())

    def test_interrupt_during_reset_uses_only_cleanup_retry(self):
        root, result, events, _ = self.run_case('reset_interrupt')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(events, ['set', 'worker', 'reset', 'reset'])
        self.assertIn('interrupted', result['reset_attempts'][0]['issues'])
        self.assertTrue(result['reset_command_succeeded'])
        self.assertFalse((root / 'lock').exists())

    def test_reset_retry_is_cleanup_only_and_initial_failure_retained(self):
        root, result, events, _ = self.run_case('reset_retry')
        self.assertEqual(events, ['set', 'worker', 'reset', 'reset'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('reset_attempt_failed:1', result['errors'])
        self.assertTrue(result['reset_command_succeeded'])
        self.assertFalse((root / 'lock').exists())

    def test_two_failed_or_timed_out_resets_alert_without_more_work(self):
        for mode in ('reset_fail', 'reset_timeout'):
            with self.subTest(mode=mode):
                root, result, events, stderr = self.run_case(mode, command_seconds=.2)
                self.assertEqual(events, ['set', 'worker', 'reset', 'reset'])
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['reset_command_succeeded'])
                self.assertTrue((root / 'lock').exists())
                self.assertIn('RESET UNCONFIRMED', stderr)

    def test_capture_exception_and_cleanup_failure_do_not_bypass_reset(self):
        for inject in ('capture', 'cleanup'):
            with self.subTest(inject=inject):
                root, result, events, _ = self.run_case('success', inject=inject)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['armed'])
                self.assertTrue(result['reset_command_succeeded'])
                self.assertNotIn('worker', events)
                self.assertFalse((root / 'lock').exists())
                if inject == 'cleanup':
                    self.assertFalse(result['subreaper_restored'])
                    self.assertEqual(events, ['set', 'reset', 'reset'])
                else:
                    self.assertEqual(events, ['reset'])

    def test_post_capture_exceptions_preserve_obligation_without_repeating_work(self):
        for inject, expected in (('set_after', ['set', 'reset']),
                                 ('worker_after', ['set', 'worker', 'reset']),
                                 ('reset_after', ['set', 'worker', 'reset', 'reset']),
                                 ('reset_clock', ['set', 'worker', 'reset'])):
            with self.subTest(inject=inject):
                root, result, events, _ = self.run_case('success', inject=inject)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(events, expected)
                self.assertTrue(result['reset_command_succeeded'])
                self.assertFalse((root / 'lock').exists())

    def test_signal_setup_or_obligation_write_failure_never_arms_or_spawns(self):
        for inject in ('setup', 'arm'):
            with self.subTest(inject=inject):
                root, result, events, _ = self.run_case('success', inject=inject)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['armed'])
                self.assertEqual(events, [])
                self.assertFalse((root / 'transaction/obligation.json').exists())

    def test_invalid_commands_env_limits_or_parent_fail_before_output(self):
        script = '''import os,sys
from pathlib import Path
from supervised_restoration import supervise_restoration
root=Path(''' + repr(str(self.root)) + ''')
base=dict(cwd=root,env={},output_dir=root/'transaction',parent_pid=os.getppid())
argv=[sys.executable,'-S','-c','raise AssertionError("must not execute")']
for changed in ({'parent_pid':0},{'env':None},{'env':{'bad=key':'x'}}, {'lock_seconds':601},
                {'command_seconds':11},{'command_seconds':True},{'lock_seconds':float('nan')},
                {'cleanup_reserve_seconds':0},{'cleanup_reserve_seconds':True},
                {'cleanup_reserve_seconds':float('inf')},{'cleanup_reserve_seconds':600},
                {'lock_seconds':5,'cleanup_reserve_seconds':6}, {'external_cancel_check':False}):
    try:supervise_restoration(argv,argv,argv,**(base|changed))
    except ValueError:pass
    else:raise AssertionError(changed)
    assert not (root/'transaction').exists()
for bad in ([],['relative'],[sys.executable,'bad\\0arg']):
    try:supervise_restoration(argv,bad,argv,**base)
    except ValueError:pass
    else:raise AssertionError(bad)
    assert not (root/'transaction').exists()
'''
        run = subprocess.run([sys.executable, '-S', '-c', script], cwd=HERE, env=ENV,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_existing_child_or_subreaper_role_is_rejected_before_outputs(self):
        script = f'''import os,sys,subprocess
from pathlib import Path
from supervised_restoration import supervise_restoration
from supervised_worker import subreaper
root=Path({str(self.root)!r})
argv=[sys.executable,'-S','-c','raise AssertionError("must not execute")']
child=subprocess.Popen([sys.executable,'-S','-c','import time;time.sleep(30)'],env={ENV!r})
try:
    try:supervise_restoration(argv,argv,argv,cwd=root,env={{}},output_dir=root/'transaction',parent_pid=os.getppid())
    except ValueError as error:assert 'exclusive fresh reaper' in str(error)
    else:raise AssertionError('existing child accepted')
    assert child.poll() is None
finally:
    child.kill();child.wait()
subreaper(1)
try:
    try:supervise_restoration(argv,argv,argv,cwd=root,env={{}},output_dir=root/'transaction',parent_pid=os.getppid())
    except ValueError as error:assert 'exclusive fresh reaper' in str(error)
    else:raise AssertionError('existing role accepted')
finally:subreaper(0)
assert not (root/'transaction').exists()
'''
        run = subprocess.run([sys.executable, '-S', '-c', script], cwd=HERE, env=ENV,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_existing_attempt_cannot_be_reused(self):
        root, result, events, _ = self.run_case('success')
        run = subprocess.run([sys.executable, '-S', '-c', code(root, 'success', os.getpid())],
                             cwd=HERE, env=ENV, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('fresh output directory', run.stderr)
        self.assertEqual((root / 'events').read_text().splitlines(), events)

    def test_parent_disappearance_still_resets_with_surviving_supervisor(self):
        template = code(self.root, 'parent_loss', 987654321, lock_seconds=5)
        driver = f'''import json,os,subprocess,sys,time
from pathlib import Path
from supervised_worker import subreaper,direct_children
subreaper(1)
code={template!r}
proxy="import os,subprocess,sys,time; from pathlib import Path; code="+repr(code)+"; p=subprocess.Popen([sys.executable,'-S','-c',code.replace('parent_pid=987654321','parent_pid='+str(os.getpid()))]); print(p.pid,flush=True)"
proxy += "\\nend=time.monotonic()+5\\nwhile not Path("+repr({str(self.root / 'worker.pid')!r})+").exists():\\n    assert p.poll() is None and time.monotonic()<end\\n    time.sleep(.01)"
p=subprocess.Popen([sys.executable,'-S','-c',proxy],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={ENV!r})
pid=int(p.stdout.readline());p.wait(timeout=7)
end=time.monotonic()+10
while True:
    waited,status=os.waitpid(pid,os.WNOHANG)
    if waited:break
    assert time.monotonic()<end
    time.sleep(.02)
stdout,stderr=p.communicate(timeout=1)
assert status==0,(stdout,stderr)
r=json.loads(Path({str(self.root / 'transaction/restoration.json')!r}).read_bytes())
assert r['status']=='failed' and r['parent_lost'] and r['reset_command_succeeded'],r
assert not Path({str(self.root / 'lock')!r}).exists()
assert direct_children()==[]
subreaper(0)
'''
        run = subprocess.run([sys.executable, '-S', '-c', driver], cwd=HERE, env=ENV,
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
