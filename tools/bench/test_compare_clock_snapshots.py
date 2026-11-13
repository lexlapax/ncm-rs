"""Supplied synthetic records only; no preparation, GPU or process execution."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import compare_clock_snapshots as check
import test_verify_clock_files as file_tests
from run_phase2_linux_comparison import plan_jobs
from verify_clock_files import declared_inventory


class SnapshotComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture=file_tests.ClockFileTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        root=self.fixture.root; contract=deepcopy(self.fixture.contract)
        taskset=contract['taskset']['path']
        previous=root/'.reference/phase2-optimization-baseline-bin/codec_bench'
        candidate=root/'target/release/examples/codec_bench'
        python=root/'.reference/venv/bin/python'
        for role,path in (('previous',previous),('candidate',candidate),('python',python)):
            contract['workloads'][role]['command_path']=str(path)
            contract['workloads'][role]['image']['path']=str(path)
        injection=dict(self.fixture.ids['injection'],path=str(root/'.reference/phase2-cupti/libcupti_trace_injection.so'))
        frozen=deepcopy(self.fixture.frozen); aliases=[]
        pinned,bound=declared_inventory(contract,frozen_identities=frozen,path_bindings=aliases,injection=injection)
        inventory=dict(library_contract=contract,frozen_identities=frozen,path_bindings=aliases,
                       declaration_admission_passed=True,launch_ready=False,inventory_completeness_verified=False)
        recheck=dict(supplied_files_rechecked=True,supplied_aliases_rechecked=True,launch_ready=False,
            inventory_completeness_verified=False,producer_authenticity_verified=False,atomic_snapshot_verified=False,
            declared_file_count=len(pinned),declared_file_bytes=sum(v['bytes'] for v in pinned.values()),
            declared_alias_count=len(bound))
        collection=root/'collection'
        jobs=plan_jobs(root/'.reference/phase2-optimization-baseline',previous.parent,collection/'sequence',
                       root=root,repeats=6,workloads=('codec',),cpu=2,taskset=taskset)
        self.snapshot=dict(schema_version=1,kind='clock-file-snapshot',project_root=str(root),collection_root=str(collection),
            candidate_head='a'*40,environment={'fixture':'same'},native={'file_identities':[injection]},
            python={'fixture':'same'},preloads={'fixture':'same'},utilities=[{'fixture':'same'}],
            tool_identities=[{'fixture':'same'}],inventory=inventory,precollection_files=recheck,jobs=jobs,
            loader={k:None for k in check.LOADER},upstream={k:None for k in check.UPSTREAM},
            timezone={k:None for k in check.TIMEZONE},
            worktrees={**{k:None for k in check.WORKTREES},'worktree_state_verified':True,
                       'states':{'accepted':{'head':'b'*40,'tracked_clean':True,'allowed_untracked_count':0},
                                 'candidate':{'head':'a'*40,'tracked_clean':True,'allowed_untracked_count':0}}})
        for key in check.FALSE_CLAIMS: self.snapshot[key]=False

    def test_equal_stable_fields_ignore_receipts_and_timestamps_without_io(self):
        current=deepcopy(self.snapshot)
        current['loader']['receipts']={'different':'pid/time'}; current['upstream']['receipts']={'different':'pid/time'}
        current['timezone']['observed_utc']='different'; current['worktrees']['states']['candidate']['allowed_untracked_count']=5
        before=deepcopy((self.snapshot,current))
        with patch('builtins.open',side_effect=AssertionError('no reads')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stats')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')):
            result=check.compare(self.snapshot,current)
        self.assertTrue(result['supplied_snapshot_fields_match']); self.assertFalse(result['launch_ready'])
        self.assertFalse(result['fresh_producer_authenticated']); self.assertEqual((self.snapshot,current),before)

    def test_drift_and_overclaims_fail_closed(self):
        for mode in ('head','env','plan','inventory','count','loader','upstream','timezone','worktree','claim'):
            with self.subTest(mode=mode):
                current=deepcopy(self.snapshot)
                if mode=='head': current['candidate_head']='b'*40
                if mode=='env': current['environment']['fixture']='drift'
                if mode=='plan': current['jobs'][0]['cwd']='/wrong'
                if mode=='inventory': current['inventory']['frozen_identities'][0]['sha256']='f'*64
                if mode=='count': current['precollection_files']['declared_file_count']+=1
                if mode=='loader': current['loader']['listings']={'drift':True}
                if mode=='upstream': current['upstream']['files']=[{'drift':True}]
                if mode=='timezone': current['timezone']['zone']='other'
                if mode=='worktree': current['worktrees']['states']['accepted']['head']='c'*40
                if mode=='claim': current['launch_ready']=True
                with self.assertRaises(ValueError): check.compare(self.snapshot,current)


if __name__ == '__main__': unittest.main()
