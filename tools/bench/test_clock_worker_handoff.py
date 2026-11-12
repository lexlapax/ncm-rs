"""Real default-budget restoration/handoff with CPU-only jobs and fake policy."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from observe_readonly_process import executable_maps
from readonly_trace_preflight import file_identity
from run_phase2_linux_comparison import plan_jobs
from test_clock_restoration_composition import POLICY, WORKLOAD

HERE = Path(__file__).resolve().parent
WORKER = '''import json,os,sys,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
from clock_worker_handoff import validate_handoff
from clock_process_observation import process_factory
import clock_codec_sequence as sequence
root=Path(sys.argv[2]);parent=int(sys.argv[3]);mode=sys.argv[4]
data=json.loads((root/'handoff.json').read_bytes())
path=root/'restoration/obligation.json'
record=json.loads(path.read_bytes())
if mode=='command':data['commands']['set'].append('unapproved')
if mode=='parent_command':data['parent_argv'].append('unexpected')
if mode=='parent_pid':parent+=1
if mode=='image':data['role_image']['inode']+=1
if mode=='environment':data['env']['HOME']='/wrong-home'
if mode=='ancestry':record['supervisor_parent_pid']+=1
if mode=='writer_pid':record['supervisor_pid']+=1
if mode=='schema':record['schema_version']=True
if mode=='work_deadline':record['work_deadline_monotonic_seconds']+=1
if mode=='expired':
    record['work_deadline_monotonic_seconds']=time.monotonic()-1
    record['reset_deadline_monotonic_seconds']=record['work_deadline_monotonic_seconds']+10
if mode=='clock':record['clock']['monotonic_before_ns']=0
if mode=='self_command':
    data['commands']['worker'].append('unexpected')
    record['worker_argv']=data['commands']['worker']
if mode in ('ancestry','writer_pid','schema','work_deadline','expired','clock','self_command'):
    path.write_text(json.dumps(record))
if mode=='duplicate':path.write_text(path.read_text().replace('"schema_version": 2','"schema_version": 2, "schema_version": 2'))
if mode=='nonfinite':path.write_text(path.read_text().replace('"lock_seconds": 600','"lock_seconds": NaN'))
if mode=='symlink':
    saved=path.with_name('saved.json');path.rename(saved);path.symlink_to(saved)
if mode=='oversize':path.write_bytes(b' '*(1024**2+1))
if mode=='public_directory':path.parent.chmod(0o755)
deadline=validate_handoff(path,parent_pid=parent,parent_argv=data['parent_argv'],commands=data['commands'],
    role_image=data['role_image'],taskset_path=data['taskset']['path'],cwd=sys.argv[1],env=data['env'])
(root/'received-deadline.json').write_text(json.dumps({'deadline':deadline,'parent_pid':parent}))
factory=process_factory(data['jobs'],env=data['env'],workloads=data['workloads'],taskset=data['taskset'],
    injection=data['injection'],libraries=data['libraries'])
with patch.object(sequence,'plan_jobs',return_value=data['jobs']):
    result=sequence.run_sequence(root/'previous',root/'binaries',root/'sequence',root=root,env=data['env'],
        work_deadline=deadline,cancel_check=lambda:False,process_observer_factory=factory,taskset=data['taskset']['path'])
raise SystemExit(0 if result['status']=='complete' else 1)
'''
RESTORER = '''import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from supervised_restoration import supervise_restoration
from supervised_worker import direct_children,subreaper
root=Path(sys.argv[2]);mode=sys.argv[3]
data=json.loads((root/'prepared.json').read_bytes())
python=data['parent_argv'][0]
policy=[python,'-S',str(root/'policy.py'),str(root)]
worker=[data['taskset']['path'],'-c','0',python,'-S',str(root/'worker.py'),sys.argv[1],str(root),str(os.getpid()),mode]
data['commands']={'set':policy+['set'],'worker':worker,'reset':policy+['reset']}
(root/'handoff.json').write_text(json.dumps(data))
if mode=='parent_affinity':os.sched_setaffinity(0,{2})
result=supervise_restoration(data['commands']['set'],worker,data['commands']['reset'],cwd=sys.argv[1],
    env=data['env'],output_dir=root/'restoration',parent_pid=os.getppid(),
    **({'lock_seconds':20} if mode=='short_budget' else {}))
assert direct_children()==[] and subreaper()==0,result
print(json.dumps(result))
raise SystemExit(0 if result['status']=='complete' else 1)
'''


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset')
                     and {0, 2} <= os.sched_getaffinity(0), 'Linux CPU0/CPU2/taskset required')
class ClockHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = str(Path(sys.executable).resolve())
        self.image = file_identity(self.python)
        self.taskset = file_identity(Path(shutil.which('taskset')).resolve())
        self.injection = file_identity(Path(shutil.which('true')).resolve())
        maps, _ = executable_maps(Path('/proc/self/maps').read_bytes())
        self.libc = file_identity(Path(next(row['path'] for row in maps
                                           if Path(row['path']).name == 'libc.so.6')).resolve())

    def run_case(self, mode):
        root = self.root / mode
        root.mkdir(mode=0o700)
        for name in ('previous', 'binaries'):
            (root / name).mkdir()
        for name, source in (('policy.py', POLICY), ('workload.py', WORKLOAD),
                             ('worker.py', WORKER), ('restorer.py', RESTORER)):
            (root / name).write_text(source)
        argv = [self.python, '-S', str(root / 'restorer.py'), str(HERE), str(root), mode]
        env = {'PATH': os.defpath, 'HOME': str(root), 'LANG': 'C', 'LC_ALL': 'C',
               'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        jobs = plan_jobs(root/'previous', root/'binaries', root/'sequence', root=root,
                         repeats=6, workloads=('codec',), cpu=2, taskset=self.taskset['path'])
        for ordinal, job in enumerate(jobs, 1):
            job['argv'] = [self.taskset['path'], '-c', '2', self.python, '-S', str(root/'workload.py'),
                           str(root), 'success', str(ordinal), job['report']]
        prepared = dict(parent_argv=argv, env=env, role_image=self.image, taskset=self.taskset,
                        injection=self.injection, jobs=jobs,
                        workloads={runtime: dict(command_path=self.python, image=deepcopy(self.image))
                                   for runtime in ('previous','candidate','python')},
                        libraries={runtime: [deepcopy(self.libc)] for runtime in ('previous','candidate','python')})
        (root/'prepared.json').write_text(json.dumps(prepared))
        run = subprocess.run([self.taskset['path'], '-c', '0', *argv], cwd=HERE, env=env,
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0 if mode == 'success' else 1, run.stdout + run.stderr)
        result = json.loads(run.stdout)
        self.assertTrue(result['reset_command_succeeded'], result)
        self.assertTrue(result['subreaper_restored'])
        self.assertFalse((root/'fake-lock').exists())
        self.assertEqual((root/'policy-events').read_text().splitlines(), ['set','reset'])
        self.assertEqual(result['worker']['returncode'], 0 if mode == 'success' else 1)
        self.assertFalse(result['hardware_policy_restoration_verified'])
        if mode != 'success':
            self.assertFalse((root/'sequence').exists())
            self.assertFalse((root/'workload-pids').exists())
            self.assertFalse((root/'received-deadline.json').exists())
        return root, result

    def test_default_budget_handoff_preserves_deadline_and_all_eighteen_real_observations(self):
        root, result = self.run_case('success')
        obligation = json.loads((root/'restoration/obligation.json').read_bytes())
        received = json.loads((root/'received-deadline.json').read_bytes())
        self.assertEqual(received['deadline'], result['work_deadline_monotonic_seconds'])
        self.assertEqual(received['deadline'], obligation['work_deadline_monotonic_seconds'])
        self.assertEqual(received['parent_pid'], result['lifetime']['pid'])
        self.assertEqual(obligation['supervisor_pid'], result['lifetime']['pid'])
        self.assertEqual(obligation['supervisor_parent_pid'], result['lifetime']['parent_pid'])
        self.assertEqual(result['reset_deadline_monotonic_seconds'] - received['deadline'], 10)
        sequence = json.loads((root/'sequence/sequence-result.json').read_bytes())
        self.assertEqual(sequence['status'], 'complete')
        self.assertEqual(len(sequence['jobs']), 18)
        self.assertEqual(len((root/'workload-pids').read_text().splitlines()), 18)
        for job in sequence['jobs']:
            self.assertEqual(job['observation']['status'], 'complete')
            self.assertFalse(Path(f'/proc/{job["receipt"]["pid"]}').exists())

    def test_live_role_or_reviewed_command_mismatch_prevents_work_but_resets(self):
        for mode in ('command','parent_command','parent_pid','parent_affinity','image','environment','self_command','ancestry','writer_pid'):
            with self.subTest(mode=mode):
                self.run_case(mode)

    def test_changed_budget_or_expired_deadline_cannot_restart_work(self):
        for mode in ('short_budget','work_deadline','expired','clock'):
            with self.subTest(mode=mode):
                self.run_case(mode)

    def test_malformed_or_nonprivate_obligation_prevents_work(self):
        for mode in ('schema','duplicate','nonfinite','symlink','oversize','public_directory'):
            with self.subTest(mode=mode):
                self.run_case(mode)


if __name__ == '__main__':
    unittest.main()
