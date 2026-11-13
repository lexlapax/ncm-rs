"""Private snapshot and procfs stubs only; no diagnostic role is executed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import clock_role_binding as binding
from clock_role_plan import role_plan


class RoleBindingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.launch=self.root/'launch'; self.launch.mkdir(mode=0o700)
        self.env={'PATH':os.defpath,'HOME':str(self.root),'LANG':'C','LC_ALL':'C',
                  'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
        self.image={'path':'/usr/bin/python3.14','device':11,'inode':12,'bytes':100,'sha256':'0'*64}
        self.snapshot={'schema_version':1,'kind':'clock-file-snapshot','project_root':str(self.root),
            'collection_root':str(self.root/'collection'),
            'tool_identities':[{'path':str(self.root/'tools/bench/clock_diagnostic_entry.py')}],
            'utilities':[self.image,{'path':'/usr/bin/nvidia-smi'},{'path':'/usr/bin/git'}],
            'inventory':{'library_contract':{'taskset':{'path':'/usr/bin/taskset'}}},'environment':self.env}
        for key in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                    'permissions_verified','restoration_verified','availability_confirmed'):
            self.snapshot[key]=False
        raw=json.dumps(self.snapshot).encode(); self.sha=hashlib.sha256(raw).hexdigest()
        path=self.launch/'snapshot.json'; path.write_bytes(raw); path.chmod(0o600)
        self.coordinator=123; self.parent=123; self.me=456
        self.coordinator_argv=['/usr/bin/python3.14','-S','/fixture/coordinator.py']
        self.gpu='GPU-11111111-1111-1111-1111-111111111111'
        self.plan=role_plan(self.snapshot,snapshot_path=path,snapshot_sha256=self.sha,launch=self.launch,
                            coordinator_pid=self.coordinator,gpu_uuid=self.gpu)
        self.role='restorer'; self.parent_argv=self.coordinator_argv
        self.parent_parent=99
        self.proc_env=b'\0'.join((k+'='+v).encode() for k,v in self.env.items())+b'\0'
        self.fakeos=SimpleNamespace(environ=self.env,getcwd=lambda:str(self.root),
            sched_getaffinity=lambda pid:{0},getppid=lambda:self.parent,getpid=lambda:self.me,
            stat=lambda path:SimpleNamespace(st_dev=11,st_ino=12))
        self.patches=[patch.object(binding,'sys',SimpleNamespace(platform='linux',flags=SimpleNamespace(no_site=1))),
                      patch.object(binding,'os',self.fakeos),patch.object(binding,'direct_children',return_value=[]),
                      patch.object(binding,'subreaper',return_value=0),patch.object(binding,'read_proc',side_effect=self.proc),
                      patch.object(binding,'process_stat',side_effect=self.stat)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)

    def proc(self,path,limit):
        if str(path)=='/proc/self/cmdline':
            argv=self.plan['observer_argv'] if self.role=='observer-supervisor' else self.plan['roles'][
                'telemetry_worker' if self.role=='telemetry' else self.role]['argv']
            return b'\0'.join(a.encode() for a in argv[3:])+b'\0'
        p=str(path).split('/'); pid=int(p[2]); field=p[3]
        if field=='stat': return f'{pid}:{self.parent_parent if pid==self.parent else 99}:77'.encode()
        if field=='environ': return self.proc_env
        if field=='cmdline':
            argv=self.parent_argv if pid==self.parent else self.coordinator_argv
            return b'\0'.join(a.encode() for a in argv)+b'\0'
        raise AssertionError(field)

    @staticmethod
    def stat(raw): return tuple(map(int,raw.decode().split(':')))

    def inspect(self):
        return binding.inspect(self.role,launch=self.launch,snapshot_sha256=self.sha,
                               coordinator_pid=self.coordinator,coordinator_argv=self.coordinator_argv,gpu_uuid=self.gpu)

    def test_direct_and_nested_parent_bindings_with_private_snapshot(self):
        for role in ('restorer','observer-supervisor','worker','telemetry'):
            with self.subTest(role=role):
                self.role=role
                if role in ('worker','telemetry'):
                    self.parent=200; self.parent_parent=self.coordinator
                    self.parent_argv=(self.plan['roles']['restorer']['argv'] if role=='worker' else self.plan['observer_argv'])[3:]
                else:
                    self.parent=self.coordinator; self.parent_parent=99; self.parent_argv=self.coordinator_argv
                value=self.inspect()
                self.assertTrue(value['sampled_role_identity_verified']); self.assertFalse(value['launch_ready'])
                self.assertFalse(value['coordinator_declaration_authenticated'])
                self.assertFalse(value['safe_ancestor_topology_verified'])

    def test_snapshot_digest_parent_command_environment_and_affinity_fail_closed(self):
        self.sha='f'*64
        with self.assertRaisesRegex(ValueError,'snapshot digest'): self.inspect()
        raw=(self.launch/'snapshot.json').read_bytes(); self.sha=hashlib.sha256(raw).hexdigest()
        self.parent_argv=['/usr/bin/python3.14','wrong']
        with self.assertRaisesRegex(ValueError,'identity differs'): self.inspect()
        self.parent_argv=self.coordinator_argv; self.proc_env=b'BAD=1\0'
        with self.assertRaisesRegex(ValueError,'identity differs'): self.inspect()
        self.proc_env=b'\0'.join((k+'='+v).encode() for k,v in self.env.items())+b'\0'
        self.fakeos.sched_getaffinity=lambda pid:{1}
        with self.assertRaisesRegex(ValueError,'affinity'): self.inspect()

    def test_snapshot_must_be_private_and_nested_parent_must_belong_to_coordinator(self):
        (self.launch/'snapshot.json').chmod(0o644)
        with self.assertRaisesRegex(ValueError,'private'): self.inspect()
        (self.launch/'snapshot.json').chmod(0o600)
        self.role='worker'; self.parent=200; self.parent_parent=999
        self.parent_argv=self.plan['roles']['restorer']['argv'][3:]
        with self.assertRaisesRegex(ValueError,'not coordinator child'): self.inspect()


class RealRoleBindingTests(unittest.TestCase):
    def test_actual_cpu0_parent_and_role_process_transport(self):
        if 0 not in os.sched_getaffinity(0) or shutil.which('taskset') is None:
            self.skipTest('CPU0 and taskset required')
        from readonly_trace_preflight import file_identity
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); launch=root/'launch'; launch.mkdir(mode=0o700)
            entry=root/'tools/bench/clock_diagnostic_entry.py'; entry.parent.mkdir(parents=True)
            tools=Path(binding.__file__).resolve().parent
            python=str(Path(sys.executable).resolve()); taskset=str(Path(shutil.which('taskset')).resolve())
            env={'PATH':os.defpath,'HOME':str(root),'LANG':'C','LC_ALL':'C',
                 'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
            gpu='GPU-11111111-1111-1111-1111-111111111111'
            snapshot={'schema_version':1,'kind':'clock-file-snapshot','project_root':str(root),
                'collection_root':str(root/'collection'),'tool_identities':[{'path':str(entry)}],
                'utilities':[file_identity(python),{'path':'/usr/bin/true'},{'path':'/usr/bin/git'}],
                'inventory':{'library_contract':{'taskset':{'path':taskset}}},'environment':env}
            for key in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                        'permissions_verified','restoration_verified','availability_confirmed'): snapshot[key]=False
            raw=json.dumps(snapshot).encode(); sha=hashlib.sha256(raw).hexdigest()
            path=launch/'snapshot.json'; path.write_bytes(raw); path.chmod(0o600)
            entry.write_text(f'''import json,sys\nsys.path.insert(0,{str(tools)!r})\nfrom clock_role_binding import inspect\na=sys.argv\nroot={str(root)!r}\ncoordinator=[{python!r},'-S',root+'/controller.py',a[7],a[5],a[11]]\nresult=inspect(a[1],launch=a[7],snapshot_sha256=a[5],coordinator_pid=int(a[9]),coordinator_argv=coordinator,gpu_uuid=a[11])\nprint(json.dumps(result))\n''')
            controller=root/'controller.py'
            controller.write_text(f'''import json,os,subprocess,sys\nsys.path.insert(0,{str(tools)!r})\nfrom clock_role_plan import role_plan\nlaunch,sha,gpu=sys.argv[1:]\nsnapshot=json.load(open(launch+'/snapshot.json'))\nplan=role_plan(snapshot,snapshot_path=launch+'/snapshot.json',snapshot_sha256=sha,launch=launch,coordinator_pid=os.getpid(),gpu_uuid=gpu)\np=subprocess.run(plan['roles']['restorer']['argv'],cwd={str(root)!r},env=dict(os.environ),capture_output=True,text=True,timeout=10)\nprint(json.dumps({{'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}))\n''')
            outer=subprocess.run([taskset,'-c','0',python,'-S',str(controller),str(launch),sha,gpu],
                                 cwd=root,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(outer.returncode,0,outer.stderr)
            result=json.loads(outer.stdout)
            self.assertEqual(result['code'],0,result['stderr'])
            observed=json.loads(result['stdout'])
            self.assertTrue(observed['sampled_role_identity_verified'])
            self.assertFalse(observed['launch_ready'])


if __name__ == '__main__': unittest.main()
