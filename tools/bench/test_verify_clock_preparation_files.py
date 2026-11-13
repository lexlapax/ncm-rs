"""Precollection file recheck with temporary local files, never hardware."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import test_verify_clock_files as files
import verify_clock_preparation_files as check
from readonly_trace_preflight import file_identity


class PreparationFilesTests(unittest.TestCase):
    def setUp(self):
        self.fixture = files.ClockFileTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.contract = deepcopy(self.fixture.contract)
        self.frozen = deepcopy(self.fixture.frozen)
        self.injection = deepcopy(self.fixture.ids['injection'])
        self.aliases = []

    def verify(self):
        return check.verify(self.contract, frozen_identities=self.frozen,
                            path_bindings=self.aliases, injection=self.injection)

    def test_rechecks_all_declared_files_and_preserves_scope(self):
        empty=self.fixture.root/'empty'; empty.write_bytes(b'')
        self.frozen.append(file_identity(empty))
        with patch('subprocess.Popen',side_effect=AssertionError('no subprocess')):
            result=self.verify()
        self.assertTrue(result['supplied_files_rechecked']); self.assertTrue(result['supplied_aliases_rechecked'])
        for key in ('inventory_completeness_verified','producer_authenticity_verified','atomic_snapshot_verified','launch_ready'):
            self.assertFalse(result[key])
        self.assertGreaterEqual(result['declared_file_count'],len(self.frozen))

    def test_bad_size_or_inode_stops_before_any_hash(self):
        path=Path(self.frozen[0]['path'])
        path.write_bytes(b'longer than the original')
        with patch.object(check,'file_identity',side_effect=AssertionError('no hash before complete inventory')):
            with self.assertRaisesRegex(ValueError,'identity changed'): self.verify()

    def test_same_size_changed_bytes_and_replacement_fail(self):
        path=Path(self.frozen[0]['path'])
        raw=path.read_bytes(); path.write_bytes(b'x'*len(raw))
        with self.assertRaisesRegex(ValueError,'digest mismatch'): self.verify()
        path.rename(path.with_name('old-source')); path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'identity changed'): self.verify()

    def test_alias_resolution_and_change_during_hash_are_rejected(self):
        root=self.fixture.root; alias=root/'extra-alias'; alias.symlink_to(self.frozen[0]['path'])
        self.aliases.append(dict(path=str(alias),canonical_path=self.frozen[0]['path']))
        self.verify()
        real=check.file_identity; changed=False
        def replace_alias(*args,**kwargs):
            nonlocal changed
            result=real(*args,**kwargs)
            if not changed:
                alias.unlink(); alias.symlink_to(self.injection['path']); changed=True
            return result
        with patch.object(check,'file_identity',side_effect=replace_alias):
            with self.assertRaisesRegex(ValueError,'alias target changed'): self.verify()

    def test_impossible_declarations_reject_before_path_access(self):
        self.frozen.extend(dict(self.frozen[0],path=str(self.fixture.root/f'extra-{i}')) for i in range(513))
        with patch.object(Path,'resolve',side_effect=AssertionError('no path access')):
            with self.assertRaisesRegex(ValueError,'count bound'): self.verify()


if __name__ == '__main__': unittest.main()
