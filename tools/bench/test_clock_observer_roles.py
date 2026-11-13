"""Observer-role wrappers with stubbed actions; no GPU or workload processes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import clock_observer_roles as roles


class ObserverRoleTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.launch=self.root/'launch'; self.launch.mkdir(mode=0o700)
        self.collection=self.root/'collection'
        self.env={'PATH':os.defpath,'HOME':str(self.root),'LANG':'C','LC_ALL':'C',
                  'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
        self.snapshot={'schema_version':1,'kind':'clock-file-snapshot','project_root':str(self.root),
            'collection_root':str(self.collection),'environment':self.env,
            'tool_identities':[{'path':str(self.root/'tools/bench/clock_diagnostic_entry.py')},
                               {'path':str(self.root/'tools/bench/sample_linux_cpu.py'),'sha256':'a'*64}],
            'utilities':[{'path':'/usr/bin/python3.14'},{'path':'/usr/bin/nvidia-smi'},{'path':'/usr/bin/git'}],
            'inventory':{'library_contract':{'taskset':{'path':'/usr/bin/taskset'}}}}
        for key in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                    'permissions_verified','restoration_verified','availability_confirmed'): self.snapshot[key]=False
        raw=json.dumps(self.snapshot).encode(); self.sha=hashlib.sha256(raw).hexdigest()
        path=self.launch/'snapshot.json'; path.write_bytes(raw); path.chmod(0o600)
        self.gpu='GPU-11111111-1111-1111-1111-111111111111'; self.coord=['/usr/bin/python3.14','-S','/fixture/root.py']
        self.patches=[patch.object(roles,'inspect_binding',side_effect=lambda role,**kw:
             {'role':role,'sampled_role_identity_verified':True,'launch_ready':False}),
             patch.object(roles,'supervise',return_value={'status':'complete'}),
             patch.object(roles,'collect',return_value={'status':'complete'}),
             patch('subprocess.Popen',side_effect=AssertionError('no child processes in wrapper tests'))]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
    def run_role(self,role,quiet=True):
        return roles.run(role,launch=self.launch,snapshot_sha256=self.sha,coordinator_pid=123,
                         coordinator_argv=self.coord,gpu_uuid=self.gpu,quiet_window_confirmed=quiet)
    def test_observer_supervisor_and_telemetry_use_frozen_plan(self):
        self.assertEqual(self.run_role('observer-supervisor')['status'],'complete')
        args=roles.supervise.call_args
        self.assertIn('telemetry',args.args[0]); self.assertEqual(args.kwargs['parent_pid'],123)
        self.assertEqual(args.kwargs['timeout_seconds'],950)
        self.assertEqual(args.kwargs['output_dir'],self.collection/'observer-supervision')
        self.assertEqual(self.run_role('telemetry')['status'],'complete')
        args=roles.collect.call_args
        self.assertEqual(args.kwargs['expected_cpu_source'],'a'*64)
        self.assertEqual(args.kwargs['output_dir'],self.collection/'telemetry')
        self.assertEqual(args.args[0][:5],['/usr/bin/taskset','-c','0,2','/usr/bin/python3.14','-S'])
        self.assertEqual(args.args[1][0],'/usr/bin/nvidia-smi')
        self.assertFalse(self.collection.exists())
    def test_missing_quiet_or_failed_binding_never_dispatches(self):
        with self.assertRaises(ValueError): self.run_role('telemetry',quiet=False)
        roles.inspect_binding.assert_not_called(); roles.collect.assert_not_called()
        roles.inspect_binding.return_value={'role':'telemetry','sampled_role_identity_verified':False,'launch_ready':False}
        roles.inspect_binding.side_effect=None
        with self.assertRaises(ValueError): self.run_role('telemetry')
        roles.collect.assert_not_called()
    def test_changed_private_snapshot_prevents_dispatch(self):
        (self.launch/'snapshot.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'digest differs'): self.run_role('observer-supervisor')
        roles.supervise.assert_not_called(); roles.collect.assert_not_called()


class RealObserverRoleTests(unittest.TestCase):
    def test_actual_observer_role_binding_before_stubbed_dispatch(self):
        if 0 not in os.sched_getaffinity(0) or shutil.which('taskset') is None:
            self.skipTest('CPU0 and taskset required')
        from readonly_trace_preflight import file_identity
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); launch=root/'launch'; launch.mkdir(mode=0o700)
            entry=root/'tools/bench/clock_diagnostic_entry.py'; entry.parent.mkdir(parents=True)
            tools=Path(roles.__file__).resolve().parent
            python=str(Path(sys.executable).resolve()); taskset=str(Path(shutil.which('taskset')).resolve())
            env={'PATH':os.defpath,'HOME':str(root),'LANG':'C','LC_ALL':'C',
                 'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
            gpu='GPU-11111111-1111-1111-1111-111111111111'
            snapshot={'schema_version':1,'kind':'clock-file-snapshot','project_root':str(root),
                'collection_root':str(root/'collection'),
                'tool_identities':[{'path':str(entry)},{'path':str(root/'tools/bench/sample_linux_cpu.py'),'sha256':'a'*64}],
                'utilities':[file_identity(python),{'path':'/usr/bin/true'},{'path':'/usr/bin/git'}],
                'inventory':{'library_contract':{'taskset':{'path':taskset}}},'environment':env}
            for key in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                        'permissions_verified','restoration_verified','availability_confirmed'): snapshot[key]=False
            raw=json.dumps(snapshot).encode(); sha=hashlib.sha256(raw).hexdigest()
            path=launch/'snapshot.json'; path.write_bytes(raw); path.chmod(0o600)
            entry.write_text(f'''import json,sys\nsys.path.insert(0,{str(tools)!r})\nimport clock_observer_roles as roles\na=sys.argv\nroot={str(root)!r}\ncoordinator=[{python!r},'-S',root+'/controller.py',a[7],a[5],a[11]]\nroles.supervise=lambda argv,**kw: {{'status':'complete','timeout':kw['timeout_seconds']}}\nresult=roles.run(a[1],launch=a[7],snapshot_sha256=a[5],coordinator_pid=int(a[9]),coordinator_argv=coordinator,gpu_uuid=a[11],quiet_window_confirmed=True)\nprint(json.dumps(result))\n''')
            controller=root/'controller.py'
            controller.write_text(f'''import json,os,subprocess,sys\nsys.path.insert(0,{str(tools)!r})\nfrom clock_role_plan import role_plan\nlaunch,sha,gpu=sys.argv[1:]\nsnapshot=json.load(open(launch+'/snapshot.json'))\nplan=role_plan(snapshot,snapshot_path=launch+'/snapshot.json',snapshot_sha256=sha,launch=launch,coordinator_pid=os.getpid(),gpu_uuid=gpu)\np=subprocess.run(plan['observer_argv'],cwd={str(root)!r},env=dict(os.environ),capture_output=True,text=True,timeout=10)\nprint(json.dumps({{'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}))\n''')
            outer=subprocess.run([taskset,'-c','0',python,'-S',str(controller),str(launch),sha,gpu],
                                 cwd=root,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(outer.returncode,0,outer.stderr)
            result=json.loads(outer.stdout); self.assertEqual(result['code'],0,result['stderr'])
            observed=json.loads(result['stdout']); self.assertEqual(observed,{'status':'complete','timeout':950})
            self.assertFalse((root/'collection').exists())


if __name__ == '__main__': unittest.main()
