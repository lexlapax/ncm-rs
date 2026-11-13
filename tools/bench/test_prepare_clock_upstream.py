"""Owned temporary Git repositories; no upstream import or GPU operations."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prepare_clock_upstream as upstream


class UpstreamPreparationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.tree = self.root/'tree'; self.tree.mkdir()
        self.git = Path('/usr/bin/git').resolve(strict=True)
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                        GIT_AUTHOR_NAME='Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid',
                        GIT_COMMITTER_NAME='Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid')
        self.run_git('init','-q')
        (self.tree/'file.txt').write_text('pinned content\n')
        (self.tree/'empty').write_bytes(b'')
        (self.tree/'.gitignore').write_text('ignored\n')
        self.run_git('add','--all')
        self.run_git('commit','-q','-m','Fixture')
        self.head = self.run_git('rev-parse','HEAD').strip()
        self.config_sha = hashlib.sha256((self.tree/'.git/config').read_bytes()).hexdigest()
        self.counter = 0

    def run_git(self,*args):
        return subprocess.check_output([str(self.git),'-C',str(self.tree),*args],env=self.env,text=True)

    def inspect(self, *, config_sha=None):
        self.counter += 1
        with patch.object(upstream,'UPSTREAM_COMMIT',self.head):
            return upstream.inspect(self.root,self.tree,self.root/f'output-{self.counter}',
                git_path=self.git,home=self.root,config_sha=config_sha or self.config_sha)

    def test_valid_tree_and_empty_blob_with_private_evidence(self):
        result = self.inspect()
        self.assertEqual(result['tracked_file_count'],3)
        self.assertTrue(result['raw_commit_sha1_verified'])
        self.assertTrue(result['replacement_objects_disabled'])
        self.assertFalse(result['launch_ready']); self.assertFalse(result['atomic_snapshot_verified'])
        self.assertEqual(next(row['identity']['bytes'] for row in result['files'] if row['relative_path']=='empty'),0)
        self.assertEqual(len(result['receipts']),10)
        self.assertEqual((self.root/'output-1').stat().st_mode & 0o777,0o700)
        self.assertTrue((self.root/'output-1/upstream.json').is_file())

    def test_drift_untracked_ignored_and_wrong_config_fail(self):
        (self.tree/'file.txt').write_text('changed\n')
        with self.assertRaises(ValueError): self.inspect()
        (self.tree/'file.txt').write_text('pinned content\n')
        (self.tree/'new').write_text('untracked')
        with self.assertRaises(ValueError): self.inspect()
        (self.tree/'new').unlink()
        (self.tree/'ignored').write_text('ignored')
        with self.assertRaisesRegex(ValueError,'ignored upstream'): self.inspect()
        (self.tree/'ignored').unlink()
        with self.assertRaises(ValueError): self.inspect(config_sha='0'*64)
        for i in range(1,5): self.assertFalse((self.root/f'output-{i}/upstream.json').exists())

    def test_wrong_pin_and_nested_output_fail(self):
        with patch.object(upstream,'UPSTREAM_COMMIT','0'*40):
            with self.assertRaises(ValueError):
                upstream.inspect(self.root,self.tree,self.root/'bad-pin',git_path=self.git,
                                 home=self.root,config_sha=self.config_sha)
        with self.assertRaises(ValueError):
            upstream.inspect(self.root,self.tree,self.tree/'inside',git_path=self.git,
                             home=self.root,config_sha=self.config_sha)
        self.assertFalse((self.tree/'inside').exists())
        with self.assertRaisesRegex(ValueError, 'configuration hash'):
            upstream.inspect(self.root,self.tree,self.root/'no-config',git_path=self.git,
                             home=self.root,config_sha=None)
        self.assertFalse((self.root/'no-config').exists())


if __name__ == '__main__': unittest.main()
