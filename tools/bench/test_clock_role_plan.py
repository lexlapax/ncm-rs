"""Pure role planning; no process, GPU utility or clock command is executed."""
from copy import deepcopy
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from clock_role_plan import role_plan


class RolePlanTests(unittest.TestCase):
    def setUp(self):
        self.root='/fixture/project'
        self.snapshot={'schema_version':1,'kind':'clock-file-snapshot','project_root':self.root,
            'collection_root':self.root+'/collection',
            'tool_identities':[{'path':self.root+'/tools/bench/clock_diagnostic_entry.py'}],
            'utilities':[{'path':'/usr/bin/python3.14'},{'path':'/usr/bin/nvidia-smi'},{'path':'/usr/bin/git'}],
            'inventory':{'library_contract':{'taskset':{'path':'/usr/bin/taskset'}}},
            'environment':{'PATH':os.defpath,'HOME':'/fixture/home','LANG':'C','LC_ALL':'C',
                           'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}}
        for key in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                    'permissions_verified','restoration_verified','availability_confirmed'):
            self.snapshot[key]=False
        self.args=dict(snapshot_path='/fixture/launch/snapshot.json',snapshot_sha256='a'*64,
                       launch='/fixture/launch',coordinator_pid=123,gpu_uuid='GPU-11111111-1111-1111-1111-111111111111')

    def plan(self): return role_plan(self.snapshot,**self.args)

    def test_pure_fixed_roles_budgets_affinity_and_policy_commands(self):
        before=deepcopy((self.snapshot,self.args))
        with patch('builtins.open',side_effect=AssertionError('no reads')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stat')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')), \
             patch('subprocess.Popen',side_effect=AssertionError('no commands')):
            result=self.plan()
        roles=result['roles']; self.assertEqual(set(roles),{'restorer','set','worker','reset','telemetry_worker','cpu'})
        self.assertEqual((roles['restorer']['timeout_seconds'],roles['restorer']['termination_grace_seconds']),(660,60))
        self.assertEqual((roles['set']['timeout_seconds'],roles['worker']['timeout_seconds'],roles['reset']['timeout_seconds']),(10,600,10))
        self.assertEqual(roles['set']['argv'][-1],'--lock-gpu-clocks=1710,1710')
        self.assertEqual(roles['reset']['argv'][-1],'--reset-gpu-clocks')
        for name in ('restorer','worker','telemetry_worker'):
            self.assertEqual(roles[name]['argv'][:5],['/usr/bin/taskset','-c','0','/usr/bin/python3.14','-S'])
            self.assertIn(self.args['gpu_uuid'],roles[name]['argv'])
        self.assertEqual(roles['cpu']['argv'][:5],['/usr/bin/taskset','-c','0,2','/usr/bin/python3.14','-S'])
        self.assertEqual(roles['cpu']['report_limit_bytes'],16*1024**2)
        self.assertFalse(result['launch_ready']); self.assertFalse(result['hardware_policy_verified'])
        result['roles']['set']['argv'].clear()
        self.assertEqual((self.snapshot,self.args),before)

    def test_rejects_scope_paths_identity_environment_and_gpu_drift(self):
        for mode in ('ready','entry','utility','env','collection','snapshot_path','sha','pid','gpu'):
            with self.subTest(mode=mode):
                self.setUp()
                if mode=='ready': self.snapshot['launch_ready']=True
                if mode=='entry': self.snapshot['tool_identities']=[]
                if mode=='utility': self.snapshot['utilities'][1]['path']='relative'
                if mode=='env': self.snapshot['environment']['LD_PRELOAD']='unreviewed'
                if mode=='collection': self.args['launch']=self.root+'/collection/nested'
                if mode=='snapshot_path': self.args['snapshot_path']='/fixture/unbound.json'
                if mode=='sha': self.args['snapshot_sha256']='bad'
                if mode=='pid': self.args['coordinator_pid']=True
                if mode=='gpu': self.args['gpu_uuid']='not-a-gpu'
                with self.assertRaises((ValueError,KeyError)): self.plan()


if __name__ == '__main__': unittest.main()
