"""Pure synthetic preparation composition; never launches a workload."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from clock_preparation_inventory import compose, upstream_group
from prepare_clock_python import UPSTREAM
import test_clock_library_contract as library_tests


class PreparationInventoryTests(unittest.TestCase):
    def setUp(self):
        fixture = library_tests.LibraryContractTests()
        fixture.setUp()
        self.fixture = fixture
        self.python, self.loader, self.preloads = deepcopy((fixture.python, fixture.loader, fixture.preloads))
        image = fixture.image
        self.python['historical'].update(declaration=image('/fixture/evidence.json'), archive=image('/fixture/evidence.tar.xz'))
        self.python['historical_source_identities'] = [image('/fixture/source.py')]
        self.python['interpreter_links'] = [dict(path='/fixture/venv/bin/python', target='/usr/bin/python3.14')]
        self.loader['loader_identity'] = image('/usr/lib/ld.so')
        self.injection = image('/fixture/injection.so')
        self.native = dict(schema_version=1, diagnostic_only=True, launch_ready=False, availability_confirmed=False,
            file_identities=deepcopy([self.injection, *(self.loader['targets'][k]['image'] for k in ('previous','candidate','taskset'))]),
            declaration_identities=[image('/fixture/native-evidence.json')])
        self.groups = {k: [image('/fixture/'+k)] for k in ('tools','utilities','timezone')}
        self.upstream = self.upstream_record()
        self.aliases = []

    def compose(self):
        return compose(self.native,self.python,self.loader,self.preloads,upstream_record=self.upstream,
                       upstream_tree='/fixture/upstream',injection=self.injection,
                       extra_groups=self.groups,path_bindings=self.aliases)

    def test_composition_is_pure_complete_for_supplied_producer_fields_and_independent(self):
        before = deepcopy((self.native,self.python,self.loader,self.preloads,self.upstream,self.groups,self.aliases))
        with patch('builtins.open', side_effect=AssertionError('no file reads')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stats')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')), \
             patch('subprocess.Popen',side_effect=AssertionError('no subprocess')):
            result = self.compose()
        paths = {v['path'] for v in result['frozen_identities']}
        for value in [*self.native['file_identities'], *self.native['declaration_identities'],
                      *self.python['historical_source_identities'], *self.python['current_only_file_identities'],
                      self.python['historical']['archive'],self.python['historical']['declaration'],
                      self.loader['loader_identity'], *(v['image'] for v in self.loader['targets'].values()),
                      *self.preloads['current_only_sources'].values(), *(v[0] for v in self.groups.values()),
                      *(v['identity'] for v in self.upstream['files']),
                      *(self.upstream[k] for k in ('git_identity','config_identity','index_identity'))]:
            self.assertIn(value['path'], paths)
        self.assertIn(dict(path='/fixture/venv/bin/python',canonical_path='/usr/bin/python3.14'), result['path_bindings'])
        self.assertTrue(result['declaration_admission_passed'])
        self.assertFalse(result['launch_ready']); self.assertFalse(result['inventory_completeness_verified'])
        result['frozen_identities'][0]['inode'] += 1
        self.assertEqual((self.native,self.python,self.loader,self.preloads,self.upstream,self.groups,self.aliases),before)

    def test_actual_new_admission_boundary_covers_large_composition(self):
        baseline = self.compose()['declared_file_count']
        self.groups['tools'] += [self.fixture.image(f'/fixture/tool-{i}') for i in range(512-baseline)]
        self.assertEqual(self.compose()['declared_file_count'],512)
        self.groups['tools'].append(self.fixture.image('/fixture/overflow'))
        with self.assertRaisesRegex(ValueError,'count bound'): self.compose()

    def test_missing_groups_and_identity_conflicts_fail_closed(self):
        for mode in ('missing','empty','conflict','injection','native','loader'):
            with self.subTest(mode=mode):
                self.setUp()
                if mode=='missing': del self.groups['tools']
                if mode=='empty': self.groups['tools'] = []
                if mode=='conflict':
                    self.groups['tools'].append(dict(self.injection, inode=self.injection['inode']+1))
                if mode=='injection': self.injection = self.fixture.image('/fixture/other-injection')
                if mode=='native': self.native['file_identities'][1]['inode'] += 1
                if mode=='loader': self.loader['targets']['python']['image']['inode'] += 1
                with self.assertRaises((ValueError,KeyError)): self.compose()

    def test_invalid_or_conflicting_alias_and_scope_rejected(self):
        for mode in ('link','alias','scope'):
            with self.subTest(mode=mode):
                self.setUp()
                if mode=='link': self.python['interpreter_links'][0]['target'] = ''
                if mode=='alias': self.aliases = [dict(path=self.python['command_path'],canonical_path=self.injection['path'])]
                if mode=='scope': self.native['launch_ready'] = True
                with self.assertRaises(ValueError): self.compose()

    def upstream_record(self):
        image = self.fixture.image
        return dict(schema_version=1, upstream_commit=UPSTREAM, current_upstream_snapshot_verified=True,
                    tracked_clean=True, replacement_objects_disabled=True, raw_commit_sha1_verified=True,
                    launch_ready=False, python_import_closure_verified=False,
                    historical_interpreter_library_bytes_verified=False, atomic_snapshot_verified=False,
                    untracked_count=0, ignored_entry_count=0, tracked_file_count=2, tracked_bytes=64,
                    git_identity=image('/usr/bin/git'), config_identity=image('/fixture/upstream/.git/config'),
                    index_identity=image('/fixture/upstream/.git/index'),
                    files=[dict(relative_path=name, git_mode='100644', git_blob_sha1='f'*40,
                                identity=dict(image('/fixture/upstream/'+name), bytes=size))
                           for name,size in (('source.py',64),('empty',0))])

    def test_upstream_record_binds_all_sources_and_git_inputs_without_io(self):
        record = self.upstream_record(); before = deepcopy(record)
        with patch('builtins.open',side_effect=AssertionError('no read')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stat')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')):
            group = upstream_group(record,tree='/fixture/upstream')
        self.assertEqual(len(group),5)
        self.assertEqual({v['path'] for v in group},
                         {v['identity']['path'] for v in record['files']} |
                         {record[k]['path'] for k in ('git_identity','config_identity','index_identity')})
        group[0]['inode'] += 1
        self.assertEqual(record,before)

    def test_upstream_record_rejects_scope_paths_and_count_drift(self):
        for mode in ('pin','flag','ignored','count','bytes','duplicate','escape','git_config','mode','blob'):
            with self.subTest(mode=mode):
                record = self.upstream_record()
                if mode=='pin': record['upstream_commit'] = '0'*40
                if mode=='flag': record['launch_ready'] = True
                if mode=='ignored': record['ignored_entry_count'] = 1
                if mode=='count': record['tracked_file_count'] += 1
                if mode=='bytes': record['tracked_bytes'] += 1
                if mode=='duplicate': record['files'][1] = deepcopy(record['files'][0])
                if mode=='escape': record['files'][0]['relative_path'] = '../escape'
                if mode=='git_config': record['config_identity']['path'] = '/fixture/other'
                if mode=='mode': record['files'][0]['git_mode'] = '120000'
                if mode=='blob': record['files'][0]['git_blob_sha1'] = 'bad'
                with self.assertRaises(ValueError): upstream_group(record,tree='/fixture/upstream')


if __name__ == '__main__': unittest.main()
