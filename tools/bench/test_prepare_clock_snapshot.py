"""Composition tests use stub producers; no GPU or workload commands."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_clock_snapshot as snapshot
from run_phase2_linux_comparison import plan_jobs


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.site=self.root/'site'; (self.site/'torch').mkdir(parents=True)
        (self.site/'torch/_C.fixture.so').write_bytes(b'elf')
        self.home=self.root; self.collection=self.root/'collection'; self.evidence=self.root/'evidence'
        self.python=Path('/usr/bin/python3').resolve(); self.git=Path('/usr/bin/git').resolve()
        self.taskset=Path('/usr/bin/taskset').resolve(); self.nvidia=Path('/usr/bin/true').resolve()
        self.previous=self.root/'.reference/phase2-optimization-baseline-bin/codec_bench'
        self.candidate=self.root/'target/release/examples/codec_bench'
        self.injection=self.root/'.reference/phase2-cupti/libcupti_trace_injection.so'
        self.ident=lambda p: dict(path=str(p),device=1,inode=2,bytes=1,sha256='0'*64)
        self.native=dict(observer_env={'PATH':'/bin','HOME':str(self.root),'LANG':'C','LC_ALL':'C','PYTHONNOUSERSITE':'1'},
                         file_identities=[self.ident(p) for p in (self.previous,self.candidate,self.injection,self.taskset)])
        self.py=dict(site_packages=str(self.site),venv_version_metadata='3.14.9',interpreter_image=self.ident(self.python))
        python_command=self.root/'.reference/venv/bin/python'
        self.libcontract=dict(workloads={r:dict(command_path=str(p)) for r,p in
                             (('previous',self.previous),('candidate',self.candidate),('python',python_command))})
        self.inventory=dict(library_contract=self.libcontract,frozen_identities=[self.ident(self.injection)],path_bindings=[])
        self.jobs=plan_jobs(self.root/'.reference/phase2-optimization-baseline',self.previous.parent,
                            self.collection/'sequence',root=self.root,repeats=6,workloads=('codec',),
                            cpu=2,taskset=str(self.taskset))
        self.patches=[
            patch.object(snapshot,'sys',SimpleNamespace(flags=SimpleNamespace(no_site=1))),
            patch.object(snapshot,'inspect_native',return_value=self.native),
            patch.object(snapshot,'verify_worktrees',return_value={'git_identity':self.ident(self.git)}),
            patch.object(snapshot,'inspect_upstream',return_value={'snapshot':'upstream'}),
            patch.object(snapshot,'inspect_python',return_value=self.py),
            patch.object(snapshot,'inspect_preloads',return_value={'snapshot':'preloads'}),
            patch.object(snapshot,'inspect_libraries',return_value={'snapshot':'loader'}),
            patch.object(snapshot,'file_identity',side_effect=lambda p:self.ident(p)),
            patch.object(snapshot,'timezone_identity',return_value={'file_identity':self.ident('/usr/share/zoneinfo/UTC'),
                                                                   'localtime_path':'/etc/localtime'}),
            patch.object(snapshot,'tool_inventory',return_value=[self.ident(self.root/'tool.py')]),
            patch.object(snapshot,'compose',return_value=self.inventory),
            patch.object(snapshot,'plan_jobs',return_value=self.jobs),
            patch.object(snapshot,'verify_files',return_value={'declared_file_count':1}),
            patch('subprocess.Popen',side_effect=AssertionError('no workload or GPU subprocess')),
        ]
        self.mocks=[p.start() for p in self.patches]
        for p in reversed(self.patches): self.addCleanup(p.stop)

    def run_prepare(self):
        return snapshot.prepare(self.root,self.collection,self.evidence,candidate_head='a'*40,home=self.home,
                                python=self.python,git=self.git,taskset=self.taskset,nvidia_smi=self.nvidia)

    def test_snapshot_requires_all_producers_and_recheck_before_exclusive_publish(self):
        import json
        result=self.run_prepare()
        data=json.loads((self.evidence/'snapshot.json').read_bytes())
        self.assertFalse(result['launch_ready']); self.assertFalse(self.collection.exists())
        for name in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                     'permissions_verified','restoration_verified','availability_confirmed'):
            self.assertFalse(data[name])
        self.assertEqual(len(data['jobs']),18)
        self.assertEqual(data['environment']['TOKENIZERS_PARALLELISM'],'false')
        self.assertTrue(snapshot.compose.called); self.assertTrue(snapshot.verify_files.called)

    def test_file_recheck_failure_never_publishes_or_creates_collection(self):
        snapshot.verify_files.side_effect=ValueError('drift')
        with self.assertRaisesRegex(ValueError,'drift'): self.run_prepare()
        self.assertFalse((self.evidence/'snapshot.json').exists()); self.assertFalse(self.collection.exists())

    def test_wrong_image_plan_fails_before_recheck_or_publication(self):
        self.jobs[-1]['argv'][3]='/unexpected'
        with self.assertRaisesRegex(ValueError,'planned command/order'): self.run_prepare()
        snapshot.verify_files.assert_not_called()
        self.assertFalse((self.evidence/'snapshot.json').exists()); self.assertFalse(self.collection.exists())

    def test_invalid_head_rejects_before_any_producer(self):
        with self.assertRaises(ValueError):
            snapshot.prepare(self.root,self.collection,self.evidence,candidate_head='bad',home=self.home,
                             python=self.python,git=self.git,taskset=self.taskset,nvidia_smi=self.nvidia)
        snapshot.inspect_native.assert_not_called(); self.assertFalse(self.evidence.exists())

    def test_independent_plan_guard_rejects_order_cwd_copies_and_argv_changes(self):
        from copy import deepcopy
        for mode in ('order','cwd','copies','argv','output'):
            with self.subTest(mode=mode):
                jobs=deepcopy(self.jobs)
                if mode=='order': jobs[0],jobs[1]=jobs[1],jobs[0]
                if mode=='cwd': jobs[0]['cwd']=str(self.root)
                if mode=='copies': jobs[1]['python_copies']=[]
                if mode=='argv': jobs[1]['argv'][4]='different.py'
                if mode=='output': jobs[0]['report']='/wrong'
                with self.assertRaises(ValueError):
                    snapshot.validate_jobs(jobs,root=self.root,collection=self.collection,
                                           taskset=self.taskset,contract=self.libcontract)


if __name__ == '__main__': unittest.main()
