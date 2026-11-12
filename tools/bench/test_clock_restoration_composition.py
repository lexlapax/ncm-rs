"""Composed real process ownership/deadlines, fake policy/workloads/observation only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
POLICY = '''import os,sys
from pathlib import Path
root=Path(sys.argv[1]);stage=sys.argv[2]
with (root/'policy-events').open('a') as file:file.write(stage+'\\n')
if stage=='set':
    assert (root/'restoration/obligation.json').is_file()
    (root/'fake-lock').touch()
else:
    pids=root/'workload-pids'
    if pids.exists():
        for pid in pids.read_text().splitlines():
            assert not Path('/proc/'+pid).exists(),('workload survives reset',pid)
    (root/'fake-lock').unlink()
'''
WORKLOAD = '''import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]; ordinal=int(sys.argv[3]); report=Path(sys.argv[4])
assert (root/'fake-lock').exists()
with (root/'workload-pids').open('a') as file:file.write(str(os.getpid())+'\\n')
if mode=='failure' and ordinal==2:raise SystemExit(7)
if mode=='observer' and ordinal==2:
    (root/'observer-failed').touch();time.sleep(30)
if mode=='deadline':time.sleep(30)
report.write_text(json.dumps({'fake':True,'ordinal':ordinal}))
time.sleep(.1)
'''
WORKER = '''import json,os,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import clock_codec_sequence as sequence
from run_phase2_linux_comparison import plan_jobs
from test_clock_codec_sequence import Observer
root=Path(sys.argv[2]);mode=sys.argv[3];taskset=sys.argv[4]
# Test transport only: production still owes bounded/authenticated role bindings.
obligation=json.loads((root/'restoration/obligation.json').read_bytes())
deadline=obligation['work_deadline_monotonic_seconds']
(root/'received-deadline.json').write_text(json.dumps({'deadline':deadline,'parent_pid':os.getppid()}))
output=root/'collection'
jobs=plan_jobs(root/'previous',root/'binaries',output,root=root,repeats=6,workloads=('codec',),cpu=2,taskset=taskset)
for ordinal,job in enumerate(jobs,1):
    job['argv']=[taskset,'-c','2',sys.executable,'-S',str(root/'workload.py'),str(root),mode,str(ordinal),job['report']]
with patch.object(sequence,'plan_jobs',return_value=jobs):
    result=sequence.run_sequence(root/'previous',root/'binaries',output,root=root,env=dict(os.environ),
        work_deadline=deadline,cancel_check=lambda:(root/'observer-failed').exists(),
        process_observer_factory=lambda *_:Observer(),taskset=taskset)
print(json.dumps({'status':result['status'],'jobs':len(result['jobs'])}),flush=True)
raise SystemExit(0 if result['status']=='complete' else 1)
'''


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset')
                     and 2 in os.sched_getaffinity(0), 'Linux with CPU2 and taskset required')
class ClockCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('previous', 'binaries'):
            (self.root / name).mkdir()
        for name, content in (('policy.py', POLICY), ('worker.py', WORKER), ('workload.py', WORKLOAD)):
            (self.root / name).write_text(content)
        self.env = {'PATH': os.defpath, 'HOME': str(self.root), 'LANG': 'C', 'LC_ALL': 'C',
                    'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}

    def run_case(self, mode):
        python = str(Path(sys.executable).resolve())
        policy = [python, '-S', str(self.root / 'policy.py'), str(self.root)]
        worker = [python, '-S', str(self.root / 'worker.py'), str(HERE), str(self.root), mode,
                  str(Path(shutil.which('taskset')).resolve())]
        lock, reserve = (1.5, .5) if mode == 'deadline' else (10, 1)
        script = f'''import json,os
from pathlib import Path
from supervised_restoration import supervise_restoration
from supervised_worker import direct_children,subreaper
root=Path({str(self.root)!r})
result=supervise_restoration({policy + ['set']!r},{worker!r},{policy + ['reset']!r},
    cwd={str(HERE)!r},env={self.env!r},output_dir=root/'restoration',parent_pid=os.getppid(),
    lock_seconds={lock},cleanup_reserve_seconds={reserve},command_seconds=1,
    external_cancel_check=lambda:(root/'observer-failed').exists())
assert direct_children()==[] and subreaper()==0,result
print(json.dumps(result))
'''
        run = subprocess.run([python, '-S', '-c', script], cwd=HERE, env=self.env,
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = json.loads(run.stdout)
        self.assertTrue(result['reset_command_succeeded'], result)
        self.assertTrue(result['subreaper_restored'])
        self.assertFalse(result['hardware_policy_restoration_verified'])
        self.assertFalse(result['benchmark_outputs_qualified'])
        self.assertFalse((self.root / 'fake-lock').exists())
        self.assertEqual((self.root / 'policy-events').read_text().splitlines(), ['set', 'reset'])
        received = json.loads((self.root / 'received-deadline.json').read_bytes())
        self.assertEqual(received['parent_pid'], result['lifetime']['pid'])
        self.assertEqual(received['deadline'], result['work_deadline_monotonic_seconds'])
        plan = json.loads((self.root / 'collection/sequence-plan.json').read_bytes())
        self.assertEqual(plan['work_deadline_monotonic_seconds'], received['deadline'])
        pids = (self.root / 'workload-pids').read_text().splitlines()
        self.assertTrue(pids)
        for pid in pids:
            self.assertFalse(Path('/proc/' + pid).exists())
        for receipt in (result['set'], result['worker'], *result['reset_attempts']):
            self.assertTrue(receipt['direct_child_reaped'])
            self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())
        return result, pids

    def test_success_runs_all_eighteen_with_shared_deadline_then_resets(self):
        result, pids = self.run_case('success')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(pids), 18)
        collection = json.loads((self.root / 'collection/sequence-result.json').read_bytes())
        self.assertEqual(collection['status'], 'complete')
        self.assertEqual(len(collection['jobs']), 18)
        self.assertFalse(collection['benchmark_outputs_qualified'])

    def test_failed_sequence_exits_nonzero_and_resets_without_later_jobs(self):
        result, pids = self.run_case('failure')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['worker']['returncode'], 1)
        self.assertEqual(len(pids), 2)
        collection = json.loads((self.root / 'collection/sequence-result.json').read_bytes())
        self.assertEqual(collection['status'], 'failed')
        self.assertEqual(len(collection['jobs']), 2)

    def test_observer_loss_cancels_worker_and_restores_unrelated_child_survives(self):
        other = subprocess.Popen([sys.executable, '-S', '-c', 'import time;time.sleep(30)'], env=self.env)
        try:
            result, pids = self.run_case('observer')
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['external_cancelled'])
            self.assertEqual(len(pids), 2)
            self.assertIsNone(other.poll())
        finally:
            other.kill(); other.wait()

    def test_shared_deadline_aborts_nested_work_and_resets_without_retry(self):
        result, pids = self.run_case('deadline')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(pids), 1)
        self.assertNotEqual(result['worker']['status'], 'complete')
        self.assertFalse(result['external_cancelled'])


if __name__ == '__main__':
    unittest.main()
