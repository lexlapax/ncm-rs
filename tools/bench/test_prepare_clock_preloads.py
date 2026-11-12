"""File-only synthetic preload inventories; no Torch import or ELF execution."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prepare_clock_preloads as preparation
from readonly_trace_preflight import file_identity


class PreloadInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root/'site'
        (self.site/'torch').mkdir(parents=True)
        (self.site/'torch/__init__.py').write_text('''raise RuntimeError('must not execute')
def _preload_cuda_deps(err=None, required=True):
    cuda_libs: list[tuple[str, str]] = [('cublas', 'libcublas.so.*[0-9]'), ('cusolver', 'libcusolver.so.*[0-9]')]
    for folder, name in cuda_libs:
        _preload_cuda_lib(folder, name, required=required)
    _preload_cuda_lib('nvtx', 'libnvToolsExt.so.*[0-9]', required=False)
''')
        (self.site/'torch/_utils_internal.py').write_text('USE_GLOBAL_DEPS = True\nUSE_RTLD_GLOBAL_WITH_LIBTORCH = False\n')
        (self.site/'torch/version.py').write_text("cuda: str = '13.2'\n__version__ = '2.15.0+cu132'\n")
        self.lib = self.site/'nvidia/cu13/lib'
        self.lib.mkdir(parents=True)
        self.image = self.lib/'libcublas.so.13'
        self.image.write_bytes(b'\x7fELFfake-never-executed')
        self.review()

    def review(self):
        self.hashes = {name: file_identity(self.site/'torch'/name)['sha256'] for name in preparation.SOURCES}

    def inventory(self):
        return preparation.inventory(self.site, reviewed_sources=self.hashes)

    def test_source_order_missing_and_scope_without_execution(self):
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('no probes')):
            result = self.inventory()
        self.assertEqual([r['folder'] for r in result['entries']], ['cublas', 'cusolver', 'nvtx'])
        self.assertEqual([len(r['matches']) for r in result['entries']], [1, 0, 0])
        self.assertTrue(all(r['runtime_selected_path'] is None for r in result['entries']))
        self.assertEqual(result['file_bytes'], self.image.stat().st_size)
        self.assertEqual(result['cuda_source_version'], '13.2')
        self.assertEqual(result['torch_source_build_tag'], '2.15.0+cu132')
        self.assertTrue(result['site_candidate_inventory_verified'])
        for key in ('runtime_selection_verified', 'complete_sys_path_verified', 'dependency_closure_verified',
                    'historical_library_bytes_verified', 'launch_ready'):
            self.assertFalse(result[key])

    def test_all_layouts_aliases_and_ambiguous_candidates_retained_not_selected(self):
        first = self.site/'nvidia/cublas/lib'
        last = self.site/'cublas/lib'
        first.mkdir(parents=True); last.mkdir(parents=True)
        (first/'libcublas.so.13').symlink_to(self.image)
        (last/'libcublas.so.14').write_bytes(b'\x7fELFother')
        result = self.inventory()
        self.assertEqual(len(result['entries'][0]['matches']), 3)
        self.assertEqual(len(result['current_only_files']), 2)
        self.assertIsNone(result['entries'][0]['runtime_selected_path'])

    def test_source_drift_and_unsupported_reviewed_declarations_fail(self):
        init = self.site/'torch/__init__.py'
        original = init.read_text()
        for replacement, rereview in ((original+'# changed', False),
                                      (original.replace("'cublas',", "'../escape',"), True),
                                      (original.replace("cuda_libs: list[tuple[str, str]] =", 'cuda_libs = compute() #'), True),
                                      (original.replace('required=False)', 'required=True)'), True),
                                      (original+'\ndef _preload_cuda_deps(): pass\n', True)):
            with self.subTest(rereview=rereview, replacement=replacement):
                init.write_text(replacement)
                if rereview: self.review()
                with self.assertRaises(ValueError): self.inventory()
                init.write_text(original); self.review()
        flags = self.site/'torch/_utils_internal.py'
        flags.write_text('USE_GLOBAL_DEPS = False\nUSE_RTLD_GLOBAL_WITH_LIBTORCH = False\n')
        self.review()
        with self.assertRaises(ValueError): self.inventory()

    def test_escape_symlinked_layout_and_non_elf_reject(self):
        outside = self.root/'outside'
        outside.write_bytes(b'\x7fELFoutside')
        raw = self.image.read_bytes()
        self.image.unlink(); self.image.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'): self.inventory()
        self.image.unlink(); self.image.write_bytes(b'not-ELF')
        with self.assertRaisesRegex(ValueError, 'not ELF'): self.inventory()
        self.image.write_bytes(raw)
        alias = self.site/'nvidia/cublas'
        alias.symlink_to(self.site/'nvidia/cu13')
        with self.assertRaisesRegex(ValueError, 'symlinked preload search'): self.inventory()

    def test_inflight_directory_and_same_byte_identity_changes_fail(self):
        actual = preparation.file_identity
        for mode in ('add', 'replace'):
            with self.subTest(mode=mode):
                changed = False
                def identity(path, *args, **kwargs):
                    nonlocal changed
                    result = actual(path, *args, **kwargs)
                    if Path(path) == self.image and not changed:
                        changed = True
                        if mode == 'add': (self.lib/'libcublas.so.14').write_bytes(b'\x7fELFnew')
                        else:
                            saved = self.root/'saved'
                            self.image.rename(saved); self.image.write_bytes(saved.read_bytes())
                    return result
                with patch.object(preparation, 'file_identity', side_effect=identity):
                    with self.assertRaises(ValueError): self.inventory()
                self.assertTrue(changed)
                (self.lib/'libcublas.so.14').unlink(missing_ok=True)

    def test_bounds_before_candidate_hashing(self):
        for name, value in (('FILE_BYTES', 1), ('TOTAL_BYTES', 1), ('MAX_FILES', 0), ('MAX_DIRECTORY_ENTRIES', 0)):
            with self.subTest(name=name), patch.object(preparation, name, value), \
                 patch.object(preparation, 'file_identity', side_effect=AssertionError('must reject before hash')):
                with self.assertRaises(ValueError): self.inventory()


if __name__ == '__main__':
    unittest.main()
