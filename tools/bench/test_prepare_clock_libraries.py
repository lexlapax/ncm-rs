"""Static-listing checks; fixtures never load Torch or initialize CUDA."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_clock_libraries as preparation
from readonly_trace_preflight import file_identity


class ClockLibraryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lib = self.root / 'libraries'
        self.lib.mkdir()
        self.image = self.elf(self.root / 'image')
        self.loader = self.elf(self.root / 'loader')
        self.libc = self.elf(self.lib / 'libc.so.6')
        self.other = self.elf(self.lib / 'libother.so.1')
        self.targets = {key: {'image': deepcopy(self.image), 'required': ['libc.so.6']}
                        for key in preparation.TARGETS}
        self.env = dict(PATH=os.defpath, HOME=str(self.root), LANG='C', LC_ALL='C', PYTHONNOUSERSITE='1')
        self.output = self.root / 'output'
        self.calls = []

    def elf(self, path):
        path.write_bytes(b'\x7fELFsynthetic-never-executed')
        path.chmod(0o700)
        return file_identity(path)

    def listing(self, libc=None):
        return f"linux-vdso.so.1 (0x123)\nlibc.so.6 => {libc or self.libc['path']} (0x456)\n"

    def probe(self, argv, cwd, env, directory, name):
        self.calls.append(name)
        self.assertEqual(argv, [self.loader['path'], '--list', self.image['path']])
        self.assertEqual(env, self.env)
        self.assertEqual(cwd, directory)
        return self.listing(), {'status': 'complete'}

    def inspect(self):
        return preparation.inspect(self.targets, loader_identity=self.loader, environment=self.env,
                                   library_roots=[self.lib], output_dir=self.output)

    def test_static_scope_and_current_only_inventory(self):
        original = deepcopy(self.targets)
        with patch.object(preparation, 'metadata_probe', side_effect=self.probe):
            result = self.inspect()
        self.assertEqual(self.calls, sorted(preparation.TARGETS))
        self.assertEqual(self.targets, original)
        self.assertEqual(set(result['current_only_libraries']), {self.libc['path']})
        self.assertEqual(result['library_bytes'], self.libc['bytes'])
        self.assertTrue(result['standalone_loader_resolution_verified'])
        for key in ('python_runtime_load_order_verified', 'runtime_loaded_libraries_verified',
                    'historical_library_bytes_verified', 'launch_ready'):
            self.assertFalse(result[key])
        self.assertEqual(json.loads((self.output/'libraries.json').read_bytes()), result)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.output/'libraries.json').stat().st_mode & 0o777, 0o600)
        with patch.object(preparation, 'metadata_probe') as probe:
            with self.assertRaises(FileExistsError): self.inspect()
            probe.assert_not_called()

    def test_prevalidation_rejects_before_any_utility_or_output(self):
        cases = ('missing', 'late_hash', 'late_inode', 'bad_required', 'env', 'root', 'loader')
        for mode in cases:
            with self.subTest(mode=mode):
                targets, env, roots, loader = deepcopy(self.targets), dict(self.env), [self.lib], deepcopy(self.loader)
                if mode == 'missing': del targets['python']
                if mode == 'late_hash': targets['torch_global_deps']['image']['sha256'] = '0'*64
                if mode == 'late_inode': targets['torch_global_deps']['image']['inode'] += 1
                if mode == 'bad_required': targets['python']['required'] = ['../libc.so.6']
                if mode == 'env': env['LD_PRELOAD'] = 'forbidden'
                if mode == 'root': roots = [Path('/')]
                if mode == 'loader': loader['bytes'] = True
                with patch.object(preparation, 'metadata_probe') as probe:
                    with self.assertRaises(ValueError):
                        preparation.inspect(targets, loader_identity=loader, environment=env,
                                            library_roots=roots, output_dir=self.output)
                    probe.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_missing_unresolved_outside_and_native_disagreement(self):
        for mode in ('missing', 'unresolved', 'outside', 'different'):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                if mode == 'missing': self.targets['candidate']['required'].append('libabsent.so')
                def probe(argv, cwd, env, directory, name):
                    if mode == 'unresolved': return 'libc.so.6 => not found\n', {}
                    if mode == 'outside': return self.listing(self.image['path']), {}
                    if mode == 'different' and name == 'previous': return self.listing(self.other['path']), {}
                    return self.listing(), {}
                with patch.object(preparation, 'metadata_probe', side_effect=probe):
                    with self.assertRaises(ValueError): self.inspect()
                self.assertFalse((self.output/'libraries.json').exists())
                self.targets['candidate']['required'] = ['libc.so.6']

    def test_failed_probe_retains_partial_directory_without_completion(self):
        def failed(argv, cwd, env, directory, name):
            (directory/'partial.receipt.json').write_text('{"status":"failed"}')
            raise ValueError('metadata command failed')
        with patch.object(preparation, 'metadata_probe', side_effect=failed):
            with self.assertRaisesRegex(ValueError, 'metadata command failed'): self.inspect()
        self.assertTrue((self.output/'partial.receipt.json').exists())
        self.assertFalse((self.output/'libraries.json').exists())

    def test_same_byte_replacement_and_link_drift_rejected(self):
        link = self.lib / 'alias.so'
        link.symlink_to(self.libc['path'])
        for mode in ('image', 'library', 'link'):
            with self.subTest(mode=mode):
                self.output = self.root / ('drift-' + mode)
                self.targets = {key: {'image': file_identity(self.image['path']), 'required': ['libc.so.6']}
                                for key in preparation.TARGETS}
                self.libc = file_identity(self.libc['path'])
                def probe(argv, cwd, env, directory, name):
                    if name == sorted(preparation.TARGETS)[-1]:
                        if mode == 'link': link.unlink(); link.symlink_to(self.other['path'])
                        else:
                            path = Path(self.image['path'] if mode == 'image' else self.libc['path'])
                            saved = self.root / ('saved-' + mode)
                            path.rename(saved); path.write_bytes(saved.read_bytes()); path.chmod(0o700)
                    return self.listing(str(link) if mode == 'link' else None), {}
                with patch.object(preparation, 'metadata_probe', side_effect=probe):
                    with self.assertRaises(ValueError): self.inspect()
                self.assertFalse((self.output/'libraries.json').exists())

    def test_inventory_bounds_are_checked_before_library_hashing(self):
        for name, limit in (('FILE_BYTES', self.image['bytes']-1), ('TOTAL_BYTES', 1), ('MAX_LIBRARIES', 0)):
            with self.subTest(name=name):
                self.output = self.root/name
                with patch.object(preparation, name, limit), patch.object(preparation, 'metadata_probe', side_effect=self.probe), \
                     patch.object(preparation, 'file_identity', side_effect=AssertionError('no library hashing')):
                    with self.assertRaises(ValueError): self.inspect()
                self.assertFalse((self.output/'libraries.json').exists())

    @unittest.skipUnless(sys.platform == 'linux' and Path('/usr/lib/ld-linux-x86-64.so.2').exists(), 'Linux glibc required')
    def test_real_loader_on_system_python_only_not_torch(self):
        image = file_identity(Path(sys.executable).resolve())
        targets = {key: {'image': image, 'required': ['libc.so.6']} for key in preparation.TARGETS}
        roots = {Path('/usr/lib').resolve(), self.root}
        prefix_lib = Path(sys.base_prefix) / 'lib'
        if prefix_lib.is_dir(): roots.add(prefix_lib.resolve())
        result = preparation.inspect(targets,
                                     loader_identity=file_identity(Path('/usr/lib/ld-linux-x86-64.so.2').resolve()),
                                     environment=self.env, library_roots=sorted(roots),
                                     output_dir=self.output)
        self.assertEqual(len(result['receipts']), 6)
        self.assertTrue(all(r['status'] == 'complete' for r in result['receipts'].values()))
        self.assertFalse(result['python_runtime_load_order_verified'])


if __name__ == '__main__':
    unittest.main()
