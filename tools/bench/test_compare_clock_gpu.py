"""Pure GPU metadata-record comparison; no subprocesses or device access."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import compare_clock_gpu as check
import test_compare_clock_snapshots as snapshot_tests


class GpuComparisonTests(unittest.TestCase):
    def setUp(self):
        uuid='GPU-11111111-1111-1111-1111-111111111111'; utility='/usr/bin/nvidia-smi'
        def receipt(argv):
            return dict(status='complete',issues=[],returncode=0,direct_child_reaped=True,capture_eof=True,
                group_kill='sent',interrupt_signal=None,pid=1,argv=argv,cwd='/fixture',timeout_seconds=10,
                output_limit_bytes=1024**2,termination_grace_seconds=0,report_limit_bytes=None,
                captured_bytes=10,observed_bytes=10,capture_sha256='a'*64)
        self.record=dict(schema_version=1,diagnostic_only=True,
            gpu=dict(uuid=uuid,pci_bus_id='00000000:01:00.0',name='Fixture GPU',visible_device_count=1,
                     historical_uuid_equality_verified=False),
            utility=dict(path=utility,device=1,inode=2,bytes=100,sha256='b'*64),
            supported_graphics_row_count=3,requested_graphics_mhz=1710,requested_graphics_in_application_list=True,
            identity_receipt=receipt([utility,'--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader']),
            supported_clocks_receipt=receipt([utility,'-i',uuid,'--query-supported-clocks=graphics','--format=csv,noheader']))
        for key in check.FALSE_CLAIMS: self.record[key]=False
    def test_matches_records_without_io_and_ignores_ephemeral_receipt_fields(self):
        current=deepcopy(self.record); current['identity_receipt']['pid']=2; current['supported_clocks_receipt']['pid']=3
        with patch('builtins.open',side_effect=AssertionError('no reads')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stats')), \
             patch('subprocess.Popen',side_effect=AssertionError('no subprocess')):
            result=check.compare(self.record,current,root='/fixture')
        self.assertTrue(result['supplied_gpu_metadata_matches']); self.assertFalse(result['launch_ready'])
        self.assertFalse(result['fresh_producer_authenticated'])
    def test_drift_invalid_receipts_and_overclaims_fail(self):
        for mode in ('uuid','pci','name','utility','count','digest','bytes','cmd','status','claim'):
            with self.subTest(mode=mode):
                current=deepcopy(self.record)
                if mode=='uuid': current['gpu']['uuid']='GPU-22222222-2222-2222-2222-222222222222'
                if mode=='pci': current['gpu']['pci_bus_id']='00000000:02:00.0'
                if mode=='name': current['gpu']['name']='Different'
                if mode=='utility': current['utility']['sha256']='c'*64
                if mode=='count': current['supported_graphics_row_count']+=1
                if mode=='digest': current['supported_clocks_receipt']['capture_sha256']='d'*64
                if mode=='bytes': current['identity_receipt']['captured_bytes']=0
                if mode=='cmd': current['supported_clocks_receipt']['argv'].append('--reset-gpu-clocks')
                if mode=='status': current['identity_receipt']['status']='failed'
                if mode=='claim': current['gpu_lock_permission_verified']=True
                with self.assertRaises(ValueError): check.compare(self.record,current,root='/fixture')

    def test_binds_exact_utility_to_file_snapshot_without_io(self):
        fixture=snapshot_tests.SnapshotComparisonTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        snapshot=deepcopy(fixture.snapshot)
        snapshot['utilities']=[{'fixture':'python'},deepcopy(self.record['utility']),{'fixture':'git'}]
        snapshot['inventory']['frozen_identities'].append(deepcopy(self.record['utility']))
        snapshot['precollection_files']['declared_file_count'] += 1
        snapshot['precollection_files']['declared_file_bytes'] += self.record['utility']['bytes']
        record=deepcopy(self.record)
        for name in ('identity_receipt','supported_clocks_receipt'):
            record[name]['cwd']=snapshot['project_root']
        with patch('builtins.open',side_effect=AssertionError('no reads')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stats')):
            result=check.bind_snapshot(snapshot,record)
        self.assertTrue(result['supplied_gpu_utility_binding_matches']); self.assertFalse(result['launch_ready'])
        snapshot['utilities'][1]['inode']+=1
        with self.assertRaisesRegex(ValueError,'utility differs'): check.bind_snapshot(snapshot,record)
        snapshot['utilities'][1]['inode']-=1
        snapshot['inventory']['frozen_identities'].pop()
        snapshot['precollection_files']['declared_file_count'] -= 1
        snapshot['precollection_files']['declared_file_bytes'] -= self.record['utility']['bytes']
        with self.assertRaisesRegex(ValueError,'not bound'): check.bind_snapshot(snapshot,record)


if __name__ == '__main__': unittest.main()
