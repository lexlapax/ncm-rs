"""Read-only preflight orchestration with stubbed producers; no GPU or policy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clock_preflight as preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.collection=self.root/'collection'; self.output=self.root/'preflight'
        self.old=self.root/'reviewed'; self.old.mkdir(mode=0o700)
        self.gpudir=self.root/'gpu-reviewed'; self.gpudir.mkdir(mode=0o700)
        self.snapshot={'project_root':str(self.root),'collection_root':str(self.collection),'candidate_head':'a'*40,
            'environment':{'HOME':str(self.root)},'utilities':[{'path':'/python'},{'path':'/nvidia'},{'path':'/git'}],
            'native':{'observer_env':{'fixture':'env'},'file_identities':[{'path':str(self.root/'.reference/phase2-cupti/libcupti_trace_injection.so')}]},
            'inventory':{'library_contract':{'taskset':{'path':'/taskset'}},'frozen_identities':[{'fixture':1}],'path_bindings':[]},
            'worktrees':{**{k:None for k in preflight.WORKTREES},
                'states':{'accepted':{'head':'b'*40,'tracked_clean':True},'candidate':{'head':'a'*40,'tracked_clean':True}}}}
        self.gpu={'gpu':{'uuid':'GPU-11111111-1111-1111-1111-111111111111','pci_bus_id':'00000000:01:00.0','name':'Fixture'}}
        self.snapshot_path=self.old/'snapshot.json'; self.gpu_path=self.gpudir/'result.json'
        self.snapshot_sha=self.write(self.snapshot_path,self.snapshot); self.gpu_sha=self.write(self.gpu_path,self.gpu)
        self.patches=[patch.object(preflight,'ROOT',self.root),patch.object(preflight,'fresh_interpreter'),
            patch.object(preflight,'validate_snapshot',side_effect=lambda x:x),patch.object(preflight,'assert_runner_image'),
            patch.object(preflight,'prepare_snapshot',side_effect=self.make_current),
            patch.object(preflight,'compare_snapshots',return_value={'supplied_snapshot_fields_match':True}),
            patch.object(preflight,'inspect_gpu',side_effect=self.make_gpu),
            patch.object(preflight,'compare_gpu',return_value={'supplied_gpu_metadata_matches':True}),
            patch.object(preflight,'bind_snapshot',return_value={'supplied_gpu_utility_binding_matches':True}),
            patch.object(preflight,'verify_worktrees',return_value=self.snapshot['worktrees']),
            patch.object(preflight,'verify_files',return_value={'supplied_files_rechecked':True,'supplied_aliases_rechecked':True,'launch_ready':False}),
            patch('subprocess.Popen',side_effect=AssertionError('no workload or GPU subprocess in stubs'))]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
    @staticmethod
    def write(path,value):
        raw=json.dumps(value).encode(); path.write_bytes(raw); path.chmod(0o600); return hashlib.sha256(raw).hexdigest()
    def make_current(self,root,collection,evidence,**kwargs):
        evidence.mkdir(mode=0o700); path=evidence/'snapshot.json'; sha=self.write(path,self.snapshot)
        return {'snapshot':str(path),'sha256':sha}
    def make_gpu(self,root,output,**kwargs):
        output.mkdir(mode=0o700)
        return self.gpu
    def run_preflight(self):
        return preflight.run(self.snapshot_path,self.snapshot_sha,self.gpu_path,self.gpu_sha,self.output)
    def test_success_runs_each_gate_without_collection_or_readiness_claim(self):
        result=self.run_preflight()
        self.assertTrue((self.output/'preflight.json').is_file()); self.assertFalse(self.collection.exists())
        self.assertTrue((self.output/'gpu/result.json').is_file())
        for key in ('collection_started','producer_authentication_verified','permissions_verified',
                    'hardware_policy_verified','restoration_verified','availability_confirmed','launch_ready'):
            self.assertFalse(result[key])
        preflight.compare_snapshots.assert_called_once(); preflight.compare_gpu.assert_called_once()
        preflight.verify_worktrees.assert_called_once(); preflight.verify_files.assert_called_once()
    def test_bad_reviewed_digest_fails_before_output_or_producers(self):
        self.snapshot_sha='f'*64
        with self.assertRaisesRegex(ValueError,'digest differs'): self.run_preflight()
        self.assertFalse(self.output.exists()); preflight.prepare_snapshot.assert_not_called()
    def test_each_gate_failure_prevents_completion_and_collection(self):
        for name in ('compare_snapshots','inspect_gpu','compare_gpu','verify_worktrees','verify_files'):
            with self.subTest(gate=name), tempfile.TemporaryDirectory(dir=self.root) as sub:
                self.output=Path(sub)/'preflight'
                mock=getattr(preflight,name); original=mock.side_effect; mock.side_effect=ValueError('gate failure')
                with self.assertRaisesRegex(ValueError,'gate failure'): self.run_preflight()
                self.assertFalse((self.output/'preflight.json').exists()); self.assertFalse(self.collection.exists())
                mock.side_effect=original
    def test_late_worktree_drift_rejected(self):
        late=dict(self.snapshot['worktrees']); late['production_base']='different'
        preflight.verify_worktrees.return_value=late
        with self.assertRaisesRegex(ValueError,'late worktree guard differs'): self.run_preflight()
        self.assertFalse((self.output/'preflight.json').exists()); preflight.verify_files.assert_not_called()
    def test_output_cannot_be_nested_in_reviewed_evidence(self):
        self.output=self.old/'nested'
        with self.assertRaisesRegex(ValueError,'disjoint'): self.run_preflight()
        self.assertFalse(self.output.exists()); preflight.prepare_snapshot.assert_not_called()


if __name__ == '__main__': unittest.main()
