"""Metadata-only fake library/Git fixtures; no model or GPU execution."""
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import readonly_trace_runtime as runtime
from readonly_trace_preflight import file_identity


class RuntimeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'metadata'
        self.env = {'PATH': os.defpath, 'HOME': str(self.root), 'LANG': 'C', 'LC_ALL': 'C',
                    'PYTHONNOUSERSITE': '1'}
        self.loader = self.root / 'loader'
        self.loader.write_text('not executable test data')
        self.paths = {}
        for name in ('accepted', 'candidate', 'injection', 'taskset'):
            path = self.root / name
            path.write_text(name)
            self.paths[name] = str(path)
        jobs = [dict(condition=c, argv=[self.paths['taskset'], '-c', '2', self.paths['accepted' if c[0] == 'A' else 'candidate']],
                     env={**self.env, **({'CUDA_INJECTION64_PATH': self.paths['injection']} if c[-1] == 'T' else {})})
                for c in ('AP', 'CP', 'AT', 'CT', 'CP', 'AT', 'CT', 'AP', 'AT', 'CT', 'AP', 'CP')]
        self.prepared = {'observer_env': self.env, 'jobs': jobs,
                         'file_identities': [file_identity(p) for p in self.paths.values()]}
        self.libraries = {}
        for name in ('libc.so.6', 'libcuda.so.1', 'libcurand.so.10', 'libcublas.so.13', 'libcupti.so.13'):
            path = self.root / name
            path.write_text(name)
            self.libraries[name] = str(path)
        self.listing = '\n'.join(f'{name} => {path} (0x123)' for name, path in self.libraries.items())

    def test_loader_parser_direct_alias_virtual_and_failures(self):
        parsed, virtual = runtime.parse_loader_listing(
            'linux-vdso.so.1 (0x12)\n' + self.listing + '\n/usr/lib/ld-linux-x86-64.so.2 (0x23)')
        self.assertEqual(parsed['ld-linux-x86-64.so.2'], '/usr/lib/ld-linux-x86-64.so.2')
        self.assertEqual(virtual, ['linux-vdso.so.1'])
        for invalid in ('', 'statically linked', 'libcuda.so.1 => not found',
                        self.listing + '\n' + self.listing.splitlines()[0],
                        self.listing + '\nwarning: unknown output',
                        self.listing + '\nlinux-vdso.so.1 (0x12)\nlinux-vdso.so.1 (0x34)'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime.parse_loader_listing(invalid)

    def test_all_targets_rechecked_and_canonical_libraries_bound(self):
        with patch.object(runtime, 'metadata_probe', return_value=(self.listing, {'status': 'complete'})) as probe:
            result = runtime.inspect_libraries(self.prepared, self.output, loader=self.loader)
        self.assertEqual(probe.call_count, 4)
        self.assertTrue(all(call.args[0][1] == '--list' for call in probe.call_args_list))
        self.assertTrue(all(call.args[2] == self.env for call in probe.call_args_list))
        self.assertEqual(len(result['libraries']), 5)
        self.assertEqual(set(result['targets']), {'accepted', 'candidate', 'injection', 'taskset'})
        self.assertTrue(result['static_loader_resolution_verified'])
        self.assertFalse(result['runtime_loaded_libraries_verified'])
        self.assertFalse(result['launch_ready'])
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(json.loads((self.output / 'libraries.json').read_text()), result)
        with self.assertRaises(FileExistsError):
            runtime.inspect_libraries(self.prepared, self.output, loader=self.loader)

    def test_modified_target_or_injected_environment_blocks_all_probes(self):
        Path(self.paths['injection']).write_text('changed')
        with patch.object(runtime, 'metadata_probe') as probe:
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                runtime.inspect_libraries(self.prepared, self.output, loader=self.loader)
            probe.assert_not_called()
        self.assertFalse(self.output.exists())
        self.prepared['observer_env']['LD_PRELOAD'] = 'unwanted'
        with self.assertRaisesRegex(ValueError, 'environment'):
            runtime.inspect_libraries(self.prepared, self.output, loader=self.loader)

    def test_missing_required_library_and_different_native_resolution_fail(self):
        for case in ('missing', 'different'):
            out = self.root / case
            changed = self.listing.replace('libcuda.so.1 =>', 'libother.so.1 =>')
            if case == 'different':
                alternative = self.root / 'alternative-libcuda'
                alternative.write_text('different')
                changed = self.listing.replace(self.libraries['libcuda.so.1'], str(alternative))
            with patch.object(runtime, 'metadata_probe', side_effect=[
                    (self.listing, {}), (changed, {}), (self.listing, {}), (self.listing, {})]):
                with self.subTest(case=case), self.assertRaises(ValueError):
                    runtime.inspect_libraries(self.prepared, out, loader=self.loader)
            self.assertFalse((out / 'libraries.json').exists())

    def test_metadata_probe_retains_small_real_cpu_child_and_failure(self):
        self.output.mkdir(mode=0o700)
        raw, receipt = runtime.metadata_probe([sys.executable, '-c', 'print("metadata fixture")'],
                                             self.root, self.env, self.output, 'fake')
        self.assertEqual(raw, 'metadata fixture\n')
        self.assertEqual(receipt['status'], 'complete')
        with self.assertRaisesRegex(ValueError, 'metadata command failed'):
            runtime.metadata_probe([sys.executable, '-c', 'raise SystemExit(2)'],
                                   self.root, self.env, self.output, 'failed')
        self.assertEqual(json.loads((self.output / 'failed.receipt.json').read_text())['returncode'], 2)

    def git_probe(self, *, candidate_status='?? .pi/private-name\0', accepted_status='',
                  head=None, changed='', manifest_changed=''):
        outputs = {'accepted-head': runtime.ACCEPTED_HEAD + '\n',
                   'candidate-head': (head or 'a' * 40) + '\n',
                   'accepted-status': accepted_status, 'candidate-status': candidate_status,
                   'production-diff': changed, 'manifest-diff': manifest_changed}
        return patch.object(runtime, 'metadata_probe', side_effect=lambda argv, cwd, env, directory, name:
                            (outputs[name], {'status': 'complete'}))

    def test_worktree_guards_and_private_session_names_not_in_summary(self):
        with self.git_probe() as probe:
            result = runtime.verify_worktrees(self.root, 'a' * 40, self.output, git=self.loader)
        self.assertEqual(probe.call_count, 6)
        self.assertEqual(result['manifest_base'], runtime.MANIFEST_HEAD)
        manifest_call = probe.call_args_list[-1].args[0]
        self.assertIn(runtime.MANIFEST_HEAD, manifest_call)
        self.assertIn(':(glob)crates/*/Cargo.toml', manifest_call)
        self.assertEqual(result['states']['candidate']['allowed_untracked_count'], 1)
        self.assertNotIn('private-name', json.dumps(result))
        self.assertTrue(result['worktree_state_verified'])
        self.assertFalse(result['launch_ready'])
        for call in probe.call_args_list:
            self.assertIn('--no-optional-locks', call.args[0])
            self.assertIn('core.fsmonitor=false', call.args[0])
            self.assertEqual(call.args[2]['GIT_CONFIG_GLOBAL'], '/dev/null')

    def test_real_temporary_git_worktrees_without_global_configuration(self):
        env = {**self.env, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
        root = self.root / 'repo'
        base = root / '.reference/phase2-optimization-baseline'
        heads = []
        for tree in (root, base):
            tree.mkdir(parents=True)
            (tree / '.gitignore').write_text('.reference/\nreports/\n')
            (tree / 'Cargo.lock').write_text('fixture only\n')
            def git(*args):
                return subprocess.check_output(['/usr/bin/git', '-C', str(tree), *args], env=env, text=True).strip()
            git('init', '-q')
            git('add', '.gitignore', 'Cargo.lock')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Fixture')
            heads.append(git('rev-parse', 'HEAD'))
        (root / '.pi').mkdir()
        (root / '.pi/private-name').write_text('not read by the verifier')
        (root / 'reports').mkdir()
        with patch.object(runtime, 'ACCEPTED_HEAD', heads[1]), \
                patch.object(runtime, 'PRODUCTION_HEAD', heads[0]), \
                patch.object(runtime, 'MANIFEST_HEAD', heads[0]):
            result = runtime.verify_worktrees(root, heads[0], root / 'reports/check')
        self.assertTrue(result['worktree_state_verified'])
        self.assertEqual(result['states']['candidate']['allowed_untracked_count'], 1)
        self.assertFalse(result['launch_ready'])

    def test_dirty_wrong_head_truncated_status_and_production_changes_rejected(self):
        changes = ({'candidate_status': ' M tracked.py\0'}, {'candidate_status': '?? unrelated\0'},
                   {'accepted_status': '?? .pi/not-allowed\0'}, {'candidate_status': 'truncated'},
                   {'head': 'b' * 40}, {'changed': 'crates/ncm-model/src/codec.rs\n'},
                   {'manifest_changed': 'crates/ncm-backend-cuda/Cargo.toml\n'})
        for index, change in enumerate(changes):
            out = self.root / f'case-{index}'
            with self.subTest(change=change), self.git_probe(**change):
                with self.assertRaises(ValueError):
                    runtime.verify_worktrees(self.root, 'a' * 40, out, git=self.loader)
            self.assertFalse((out / 'worktrees.json').exists())


if __name__ == '__main__':
    unittest.main()
