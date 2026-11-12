"""Synthetic declarations only; compiler/factory must perform no file/probe work."""
from copy import deepcopy
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from clock_library_contract import compile_contract, CORE_TORCH, NATIVE
from clock_process_observation import process_factory, ORDER


class LibraryContractTests(unittest.TestCase):
    def setUp(self):
        self.counter = 100
        self.site = '/fixture/site'
        self.static = {}
        def dependency(name, base='/usr/lib'):
            path = base+'/'+name
            if path not in self.static: self.static[path] = self.image(path)
            return {'listed_path': path, 'canonical_path': path}
        native = {n: dependency(n) for n in sorted(NATIVE)}
        py = {n: dependency(n) for n in ('libc.so.6', 'libm.so.6', 'libpython3.14.so.1.0', 'ld-linux-x86-64.so.2')}
        core = {n: dependency(n, self.site+'/torch/lib') for n in CORE_TORCH}
        core['libc.so.6'] = native['libc.so.6']
        cudart = dependency('libcudart.so.13', self.site+'/nvidia/cu13/lib')
        self.cudart = cudart['canonical_path']
        images = {k: self.image('/fixture/'+k+'/codec_bench') for k in ('previous','candidate')}
        images.update(python=self.image('/usr/bin/python3.14'), taskset=self.image('/usr/bin/taskset'),
                      torch_extension=self.image(self.site+'/torch/_C.cpython-314-x86_64-linux-gnu.so'),
                      torch_global_deps=self.image(self.site+'/torch/lib/libtorch_global_deps.so'))
        self.loader = dict(schema_version=1, launch_ready=False, historical_library_bytes_verified=False,
                           standalone_loader_resolution_verified=True, python_runtime_load_order_verified=False,
                           runtime_loaded_libraries_verified=False,
                           targets={k: {'image': v} for k,v in images.items()},
                           listings={k: {'dependencies': deepcopy(v)} for k,v in {
                               'previous':native,'candidate':native,'python':py,'torch_extension':core,
                               'torch_global_deps':{'libc.so.6':native['libc.so.6'],'libcudart.so.13':cudart},
                               'taskset':{'libc.so.6':native['libc.so.6']}}.items()},
                           current_only_libraries=self.static, library_bytes=sum(v['bytes'] for v in self.static.values()))
        sources = {n:self.image(self.site+'/torch/'+n) for n in ('__init__.py','_utils_internal.py','version.py')}
        self.python = dict(schema_version=1, launch_ready=False, historical_library_bytes_verified=False,
                           site_packages=self.site, venv_version_metadata='3.14.9',
                           historical={'python_version':'Python 3.14.9','report':{'torch':'2.15.0+cu132'}},
                           torch_source_build_tag='2.15.0+cu132', interpreter_image=deepcopy(images['python']),
                           command_path='/fixture/venv/bin/python',
                           current_only_file_identities=[deepcopy(sources['version.py']), deepcopy(images['python'])])
        for k in ('historical_interpreter_bytes_verified','runtime_version_probed','torch_imported',
                  'startup_import_closure_verified','library_resolution_verified','current_upstream_worktree_verified'):
            self.python[k] = False
        extra = self.image(self.site+'/nvidia/cu13/lib/libcusolver.so.12')
        self.extra = extra['path']
        candidates = {self.cudart:deepcopy(self.static[self.cudart]), self.extra:extra}
        self.preloads = dict(schema_version=1, launch_ready=False, historical_library_bytes_verified=False,
                             site_candidate_inventory_verified=True, runtime_selection_verified=False,
                             complete_sys_path_verified=False, dependency_closure_verified=False,
                             site_packages=self.site, torch_source_build_tag='2.15.0+cu132',
                             current_only_sources=sources, current_only_files=candidates,
                             file_bytes=sum(v['bytes'] for v in candidates.values()),
                             entries=[dict(ordinal=i, runtime_selected_path=None,
                                           matches=[{'path':p,'canonical_path':p}] if p else [])
                                      for i,p in enumerate((self.cudart,self.extra,None))])

    def image(self, path):
        self.counter += 1
        return dict(path=path, device=29, inode=self.counter, bytes=64, sha256=f'{self.counter:064x}')

    def compile(self): return compile_contract(self.python, self.loader, self.preloads)

    def test_pure_selection_and_factory_compatibility(self):
        with patch('builtins.open',side_effect=AssertionError('no read')), \
             patch.object(Path,'open',side_effect=AssertionError('no read')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')), \
             patch('os.stat',side_effect=AssertionError('no stat')), \
             patch('hashlib.sha256',side_effect=AssertionError('no hash')), \
             patch('subprocess.Popen',side_effect=AssertionError('no subprocess')):
            result = self.compile()
            jobs = [dict(runtime=r,repeat=i//3+1,workload='codec',
                         argv=[result['taskset']['path'],'-c','2',result['workloads'][r]['command_path'],'unchanged'])
                    for i,r in enumerate(ORDER)]
            env = dict(PATH=os.defpath, HOME='/fixture/home', LANG='C', LC_ALL='C',
                       PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false')
            factory = process_factory(jobs,env=env,injection=self.image('/fixture/forbidden.so'),
                                      **{k:result[k] for k in ('workloads','taskset','libraries','known_libraries')})
            observers = [factory(i,j,env) for i,j in enumerate(jobs,1)]
        self.assertEqual(len(observers),18)
        self.assertEqual({k:len(v) for k,v in result['libraries'].items()}, {'previous':4,'candidate':4,'python':10})
        self.assertIn(self.extra, {v['path'] for v in result['known_libraries']})
        self.assertTrue(all(self.extra not in {v['path'] for v in values} for values in result['libraries'].values()))
        self.assertEqual(result['workloads']['python']['command_path'], '/fixture/venv/bin/python')
        self.assertEqual(result['workloads']['python']['image']['path'], '/usr/bin/python3.14')
        self.assertTrue(result['prepared_declaration_consistency_verified'])
        self.assertFalse(result['runtime_mapping_verified']); self.assertFalse(result['launch_ready'])

    def test_output_and_input_states_are_independent(self):
        result = self.compile()
        wanted = deepcopy(result)
        self.preloads['current_only_files'][self.extra]['inode'] += 1
        self.loader['targets']['python']['image']['inode'] += 1
        self.assertEqual(result,wanted)
        result['libraries']['previous'][0]['inode'] += 1
        self.assertEqual(result['libraries']['candidate'],wanted['libraries']['candidate'])
        self.assertEqual(result['known_libraries'],wanted['known_libraries'])

    def test_cross_record_identity_build_source_and_namespace_drift(self):
        for mode in ('image','source','build','site','overlap','observed_fields','image_inventory'):
            with self.subTest(mode=mode):
                py,ld,pre = deepcopy((self.python,self.loader,self.preloads))
                if mode=='image': py['interpreter_image']['inode'] += 1
                if mode=='image_inventory': py['current_only_file_identities'].pop()
                if mode=='source': pre['current_only_sources']['version.py']['sha256'] = 'f'*64
                if mode=='build': pre['torch_source_build_tag'] = 'different'
                if mode=='site': pre['site_packages'] = '/elsewhere'
                if mode=='overlap': pre['current_only_files'][self.cudart]['inode'] += 1
                if mode=='observed_fields': pre['current_only_files'][self.extra]['path_inode'] = 1
                with self.assertRaises(ValueError): compile_contract(py,ld,pre)

    def test_missing_dependencies_unbound_candidates_and_conflicting_aliases(self):
        for mode in ('core','native','native_difference','missing_identity','unused','selected','alias','bytes'):
            with self.subTest(mode=mode):
                py,ld,pre = deepcopy((self.python,self.loader,self.preloads))
                if mode=='core':
                    dep = ld['listings']['torch_extension']['dependencies'].pop(CORE_TORCH[0])
                    del ld['current_only_libraries'][dep['canonical_path']]
                    ld['library_bytes'] = sum(v['bytes'] for v in ld['current_only_libraries'].values())
                if mode=='native':
                    dep = ld['listings']['previous']['dependencies'].pop('libcurand.so.10')
                    del ld['listings']['candidate']['dependencies']['libcurand.so.10']
                    del ld['current_only_libraries'][dep['canonical_path']]
                    ld['library_bytes'] = sum(v['bytes'] for v in ld['current_only_libraries'].values())
                if mode=='native_difference': del ld['listings']['candidate']['dependencies']['libcurand.so.10']
                if mode=='missing_identity': del ld['current_only_libraries'][self.cudart]
                if mode=='unused': pre['entries'][1]['matches'] = []
                if mode=='selected': pre['entries'][1]['runtime_selected_path'] = self.extra
                if mode=='alias': pre['entries'][1]['matches'][0]['path'] = self.cudart
                if mode=='bytes': pre['file_bytes'] = True
                with self.assertRaises(ValueError): compile_contract(py,ld,pre)

    def test_scope_schema_and_combined_budget_not_promoted(self):
        for mode in ('schema','loader','preload','python','budget','oversized'):
            with self.subTest(mode=mode):
                py,ld,pre = deepcopy((self.python,self.loader,self.preloads))
                if mode=='schema': pre['schema_version'] = True
                if mode=='loader': ld['runtime_loaded_libraries_verified'] = True
                if mode=='preload': pre['complete_sys_path_verified'] = True
                if mode=='python': py['launch_ready'] = True
                if mode=='budget':
                    for v in ld['current_only_libraries'].values(): v['bytes'] = 512*1024**2
                    ld['library_bytes'] = sum(v['bytes'] for v in ld['current_only_libraries'].values())
                    pre['current_only_files'][self.cudart]['bytes'] = 512*1024**2
                    pre['current_only_files'][self.extra]['bytes'] = 2*1024**3
                    pre['file_bytes'] = sum(v['bytes'] for v in pre['current_only_files'].values())
                if mode=='oversized':
                    pre['current_only_files'][self.extra]['bytes'] = 2*1024**3+1
                    pre['file_bytes'] = sum(v['bytes'] for v in pre['current_only_files'].values())
                with self.assertRaises(ValueError): compile_contract(py,ld,pre)


if __name__ == '__main__': unittest.main()
