"""Dedicated-process subreaper tests with fake CPU workers only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bounded_child import run_bounded
import supervised_worker as supervisor

ENV = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'}
HERE = Path(__file__).parent


def supervisor_code(root, worker, parent, timeout=3):
    return ('import json,sys; from supervised_worker import supervise,subreaper,direct_children; '
            f'r=supervise([sys.executable,"-c",{worker!r}], cwd={str(root)!r}, '
            f'env={ENV!r}, output_dir={str(root / "supervisor")!r}, '
            f'timeout_seconds={timeout!r}, parent_pid={parent!r}); '
            'assert subreaper()==0 and direct_children()==[]; print(json.dumps(r))')


def escaped_child_worker(ending):
    return ('import os,sys,subprocess,signal,time; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"], '
            'start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); '
            'print(p.pid,flush=True); ' + ending)


@unittest.skipUnless(sys.platform == 'linux', 'Linux subreaper ownership contract')
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_supervisor(self, worker, timeout=3):
        run = subprocess.run([sys.executable, '-c', supervisor_code(self.root, worker, os.getpid(), timeout)],
                             cwd=HERE, env=ENV, text=True, capture_output=True, timeout=15, check=True)
        result = json.loads(run.stdout)
        self.assertEqual(result, json.loads((self.root / 'supervisor/supervisor.json').read_text()))
        self.assertTrue(result['cleanup']['complete'])
        self.assertTrue(result['subreaper_restored'])
        for entry in result['cleanup']['reaped']:
            self.assertFalse(Path(f'/proc/{entry["pid"]}').exists())
        return result

    def test_success_and_nonzero(self):
        result = self.run_supervisor('print("done")')
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['launch_provenance_verified'])
        self.assertFalse(result['benchmark_outputs_qualified'])
        self.assertEqual(result['cleanup']['reaped'], [])
        self.root = self.root / 'nonzero'
        self.root.mkdir()
        result = self.run_supervisor('raise SystemExit(7)')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['worker']['returncode'], 7)

    def test_worker_crash_and_normal_exit_clean_escaped_orphans(self):
        for label, ending in (('normal', 'os._exit(0)'), ('crash', 'os.kill(os.getpid(),signal.SIGKILL)')):
            root = self.root
            self.root = root / label
            self.root.mkdir()
            result = self.run_supervisor(escaped_child_worker(ending))
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['orphan_observed'])
            self.assertEqual(len(result['cleanup']['reaped']), 1)
            self.root = root

    def test_orphan_while_worker_lives_cancels_before_timeout(self):
        spawning = escaped_child_worker('os._exit(0)')
        worker = ('import subprocess,sys,time; '
                  f'p=subprocess.Popen([sys.executable,"-c",{spawning!r}], '
                  'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); p.wait(); time.sleep(30)')
        result = self.run_supervisor(worker, timeout=5)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['orphan_observed'])
        self.assertIn('external_cancelled', result['worker']['issues'])
        self.assertEqual(len(result['cleanup']['reaped']), 1)

    def test_multi_generation_session_escape_is_adopted_and_reaped(self):
        ready = self.root / 'ready'
        child = (escaped_child_worker(f'open({str(ready)!r},"w").write("ready"); time.sleep(30)'))
        worker = ('import os,sys,subprocess,time; from pathlib import Path; '
                  f'subprocess.Popen([sys.executable,"-c",{child!r}], start_new_session=True, '
                  'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
                  f'\nwhile not Path({str(ready)!r}).exists(): time.sleep(.01)\nos._exit(0)')
        result = self.run_supervisor(worker)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['cleanup']['reaped']), 2)

    def test_unrelated_test_owned_child_survives_orphan_cleanup(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], env=ENV)
        try:
            result = self.run_supervisor(escaped_child_worker('os._exit(0)'))
            self.assertEqual(result['status'], 'failed')
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.kill()
            unrelated.wait()

    def test_setup_capture_and_cleanup_failures_are_not_complete(self):
        for case in ('setup', 'capture', 'cleanup', 'restore'):
            root = self.root / case
            root.mkdir()
            script = f'''
import json,os,signal
from unittest.mock import patch
import supervised_worker as s
actual=signal.signal
actual_subreaper=s.subreaper
case={case!r}
def subreaper(value=None):
    if value==0 and case=='restore': raise OSError('simulated restoration failure')
    return actual_subreaper(value)
def handler(sig, value):
    if sig==signal.SIGTERM and case=='setup': raise OSError('simulated handler setup failure')
    return actual(sig,value)
with patch.object(s.signal,'signal',side_effect=handler), patch.object(s,'subreaper',side_effect=subreaper):
    with patch.object(s,'run_bounded',side_effect=OSError('simulated capture failure')) if case=='capture' else patch.object(s,'run_bounded',return_value={{'status':'complete'}}):
        with patch.object(s,'cleanup_owned',return_value={{'complete':False,'errors':['simulated cleanup failure']}}) if case=='cleanup' else patch.object(s,'cleanup_owned',wraps=s.cleanup_owned):
            result=s.supervise(['never executed'],cwd={str(root)!r},env={ENV!r},output_dir={str(root / 'supervisor')!r},timeout_seconds=1,parent_pid={os.getpid()})
assert result['status']=='failed',result
assert result['subreaper_restored'] is (case not in ('cleanup','restore')),result
assert s.direct_children()==[]
s.subreaper(0)
print(json.dumps({{'failed':True}}))
'''
            with self.subTest(case=case):
                run = subprocess.run([sys.executable, '-c', script], cwd=HERE, env=ENV,
                                     text=True, capture_output=True, check=True, timeout=10)
                self.assertEqual(json.loads(run.stdout), {'failed': True})

    def test_timeout_and_interrupt(self):
        result = self.run_supervisor(escaped_child_worker('time.sleep(30)'), timeout=0.15)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('child_timeout', result['worker']['issues'])
        self.root = self.root / 'interrupt'
        self.root.mkdir()
        result = self.run_supervisor('import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('interrupted', result['worker']['issues'])

    def test_existing_child_and_parent_mismatch_rejected_without_outputs(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], env=ENV)
        try:
            with self.assertRaisesRegex(ValueError, 'fresh supervisor'):
                supervisor.supervise(['unused'], cwd=self.root, env=ENV, output_dir=self.root / 'bad',
                                     timeout_seconds=1, parent_pid=os.getppid())
            self.assertIsNone(child.poll())
            with self.assertRaisesRegex(ValueError, 'parent identity'):
                supervisor.supervise(['unused'], cwd=self.root, env=ENV, output_dir=self.root / 'bad',
                                     timeout_seconds=1, parent_pid=0)
            self.assertFalse((self.root / 'bad').exists())
        finally:
            child.kill()
            child.wait()

    def test_external_cancel_before_spawn_and_during_capture(self):
        for before in (True, False):
            capture, receipt = self.root / str(before), self.root / (str(before) + '.json')
            predicate = (lambda: True) if before else iter([False, True]).__next__
            with patch('bounded_child.subprocess.Popen', wraps=subprocess.Popen) as spawn:
                result = run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     cwd=self.root, env=ENV, capture_path=capture, receipt_path=receipt,
                                     timeout_seconds=2, output_limit_bytes=100, cancel_check=predicate)
            self.assertEqual(spawn.call_count, 0 if before else 1)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('external_cancelled', result['issues'])

    def test_lost_cleanup_ownership_never_signals_numeric_pid(self):
        # No real child created; emulate an invalid caller with another reaper.
        with patch.object(supervisor, 'direct_children', side_effect=[[123], []]), \
                patch.object(supervisor.os, 'waitid', side_effect=ChildProcessError), \
                patch.object(supervisor.os, 'kill') as kill:
            result = supervisor.cleanup_owned()
        kill.assert_not_called()
        self.assertFalse(result['complete'])
        self.assertIn('child_ownership_unconfirmed', result['errors'])

    def test_parent_disappearance_is_detected_and_reaped_by_test_driver(self):
        # A separate driver adopts the orphan supervisor so this test leaks no zombies.
        worker = escaped_child_worker('time.sleep(30)')
        code_template = supervisor_code(self.root, worker, 987654321, timeout=10)
        driver = f'''
import json,os,signal,subprocess,sys,time
from supervised_worker import subreaper
subreaper(1)
code={code_template!r}
proxy_code="import os,subprocess,sys,time; code="+repr(code)+"; p=subprocess.Popen([sys.executable,'-c',code.replace('parent_pid=987654321','parent_pid='+str(os.getpid()))]); print(p.pid,flush=True); ready="+repr({str(self.root / 'supervisor/worker.txt')!r})
proxy_code += "\\nend=time.monotonic()+5\\nwhile not (os.path.exists(ready) and os.stat(ready).st_size):\\n    if p.poll() is not None or time.monotonic()>end: raise RuntimeError('worker did not become ready')\\n    time.sleep(.01)"
p=subprocess.Popen([sys.executable,'-c',proxy_code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
pid=int(p.stdout.readline())
p.wait(timeout=8)
end=time.monotonic()+12
while True:
    waited,status=os.waitpid(pid,os.WNOHANG)
    if waited: break
    if time.monotonic()>end: raise AssertionError('supervisor did not finish')
    time.sleep(.02)
result=json.loads(open({str(self.root / 'supervisor/supervisor.json')!r}).read())
assert result['status']=='failed' and result['parent_lost'] and result['cleanup']['complete'],result
assert not os.path.exists('/proc/'+str(pid))
subreaper(0)
p.stdout.close(); p.stderr.close()
print(json.dumps({{'parent_lost':True,'cleanup':True}}))
'''
        run = subprocess.run([sys.executable, '-c', driver], cwd=HERE, env=ENV,
                             text=True, capture_output=True, check=True, timeout=20)
        self.assertEqual(json.loads(run.stdout), {'parent_lost': True, 'cleanup': True})


if __name__ == '__main__':
    unittest.main()
