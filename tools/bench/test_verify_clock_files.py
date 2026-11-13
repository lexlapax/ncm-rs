"""Synthetic files and CPU-child/system-library transport; no Torch or GPU work."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import time
import tempfile
import unittest
from unittest.mock import patch

from readonly_trace_preflight import file_identity
import verify_clock_files as check
import verify_clock_processes
import test_clock_process_observation as live


class ClockFileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.lib = self.root/'lib'; self.lib.mkdir()
        self.ids = {}
        for name in ('previous','candidate','python','taskset','injection','required.so','optional.so','lazy.so','source'):
            path = (self.lib if name.endswith('.so') else self.root)/name
            path.write_bytes((b'\x7fELF' if name != 'source' else b'text')+name.encode())
            self.ids[name] = file_identity(path)
        self.alias = self.lib/'alias.so'; self.alias.symlink_to('required.so')
        self.command = self.root/'python-command'; self.command.symlink_to('python')
        self.contract = dict(schema_version=1,prepared_declaration_consistency_verified=True,
            runtime_mapping_verified=False,launch_ready=False,taskset=self.ids['taskset'],
            workloads={r:dict(image=self.ids[r],command_path=str(self.command) if r=='python' else self.ids[r]['path'])
                       for r in ('previous','candidate','python')},
            libraries={r:[self.ids['required.so']] for r in ('previous','candidate','python')},
            known_libraries=[self.ids['required.so'], self.ids['optional.so']],
            path_bindings=[dict(path=str(self.alias),canonical_path=self.ids['required.so']['path'])])
        self.observed = {}
        for name in ('previous','candidate','python','required.so','lazy.so'):
            value = self.ids[name]
            self.observed[value['path']] = dict(path=value['path'],device=28,inode=value['inode'],
                path_device=value['device'],path_inode=value['inode'],kernel_map_link_verified=True)
        self.binding = dict(schema_version=1,process_receipt_consistency_verified=True,
            frozen_files_rechecked=False,mapped_file_bytes_verified=False,launch_provenance_verified=False,
            observed_files=self.observed)
        self.frozen = [self.ids['source']]

    def verify(self, **kw):
        args = dict(frozen_identities=self.frozen,path_bindings=[],injection=self.ids['injection'],library_roots=[self.lib])
        args.update(kw)
        return check.verify_files(self.binding,self.contract,**args)

    def declare(self, **kw):
        args=dict(frozen_identities=self.frozen,path_bindings=[],injection=self.ids['injection'])
        args.update(kw)
        return check.declared_inventory(self.contract,**args)

    def test_pure_admission_count_boundary_deduplication_and_post_only_overflow(self):
        self.assertEqual(check.FILE_COUNT_LIMIT,512)
        frozen=deepcopy(self.frozen)
        frozen += [dict(self.ids['source'],path=str(self.root/f'future-{i}')) for i in range(504)]
        frozen.append(deepcopy(self.ids['source']))  # Equal repeated declarations count once.
        before=deepcopy((self.contract,frozen))
        with patch('builtins.open',side_effect=AssertionError('no reads')), \
             patch.object(os,'open',side_effect=AssertionError('no descriptor reads')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stats')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')), \
             patch.object(check,'file_identity',side_effect=AssertionError('no hashing')):
            pinned,aliases=self.declare(frozen_identities=frozen)
            self.assertEqual(len(pinned),512); self.assertEqual(len(aliases),4)
            for count in (261,303):
                admitted,_=self.declare(frozen_identities=frozen[:count-7])
                self.assertEqual(len(admitted),count)
            for count in (513, 514):
                oversized=frozen+[dict(self.ids['source'],path=str(self.root/f'extra-{i}')) for i in range(count-512)]
                with self.subTest(count=count), self.assertRaisesRegex(ValueError,'count bound'):
                    self.declare(frozen_identities=oversized)
            # The post-only lazy mapping is a 513th path: early admission never
            # replaces the final union limit or reserves room for unknown mappings.
            with self.assertRaisesRegex(ValueError,'count bound'): self.verify(frozen_identities=frozen)
        pinned[self.ids['source']['path']]['sha256']='0'*64; aliases.clear()
        self.assertEqual((self.contract,frozen),before)

    def test_pure_admission_byte_boundaries_without_reading_large_files(self):
        self.assertEqual((check.FILE_SIZE_LIMIT,check.FILE_BYTES_LIMIT),(2*1024**3,8*1024**3))
        baseline,_=self.declare(); size=sum(v['bytes'] for v in baseline.values())
        extra=[dict(self.ids['source'],path=str(self.root/f'large-{i}'),
                    bytes=check.FILE_SIZE_LIMIT-(size if i==3 else 0)) for i in range(4)]
        with patch.object(Path,'stat',side_effect=AssertionError('no stats')), \
             patch.object(Path,'resolve',side_effect=AssertionError('no resolution')), \
             patch.object(check,'file_identity',side_effect=AssertionError('no hashing')):
            pinned,_=self.declare(frozen_identities=self.frozen+extra)
            self.assertEqual(sum(v['bytes'] for v in pinned.values()),8*1024**3)
            extra[-1]['bytes']+=1
            with self.assertRaisesRegex(ValueError,'byte bound'): self.declare(frozen_identities=self.frozen+extra)
            extra[-1]['bytes']-=1; extra[0]['bytes']+=1
            with self.assertRaisesRegex(ValueError,'per-file bound'): self.declare(frozen_identities=self.frozen+extra)

    def test_empty_frozen_source_is_rehashed_but_empty_executable_is_rejected(self):
        empty = self.root/'empty-source'; empty.write_bytes(b'')
        frozen = self.frozen + [file_identity(empty)]
        pinned, _ = self.declare(frozen_identities=frozen)
        self.assertEqual(pinned[str(empty)]['bytes'], 0)
        result = self.verify(frozen_identities=frozen)
        row = next(v for v in result['files'] if v['identity']['path'] == str(empty))
        self.assertEqual(row['identity']['sha256'], frozen[-1]['sha256'])
        empty.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity changed'): self.verify(frozen_identities=frozen)
        contract = deepcopy(self.contract)
        contract['workloads']['previous']['image']['bytes'] = 0
        with self.assertRaises(ValueError):
            check.declared_inventory(contract, frozen_identities=self.frozen,
                                     path_bindings=[], injection=self.ids['injection'])

    def test_loader_alias_dotdot_resolves_through_symlink_parent_without_lexical_collapse(self):
        (self.lib/'nested').mkdir()
        link = self.root/'loader-link'; link.symlink_to(self.lib/'nested')
        spelling = str(link)+'/../required.so'
        self.contract['path_bindings'].append(dict(path=spelling, canonical_path=self.ids['required.so']['path']))
        _, aliases = self.declare()
        self.assertEqual(aliases[spelling], self.ids['required.so']['path'])
        self.verify()
        other = self.root/'other'; other.mkdir(); (other/'nested').mkdir()
        (other/'required.so').write_bytes(b'other')
        link.unlink(); link.symlink_to(other/'nested')
        with self.assertRaisesRegex(ValueError, 'path alias target changed'):
            self.verify()
        with self.assertRaisesRegex(ValueError, 'absolute normalized path'):
            self.declare(path_bindings=[dict(path=spelling, canonical_path=self.ids['required.so']['path'])])

    def test_pure_admission_alias_boundary_and_conflicts_without_resolution(self):
        self.assertEqual(check.PATH_COUNT_LIMIT,512)
        aliases=[dict(path=str(self.root/f'alias-{i}'),canonical_path=self.ids['source']['path']) for i in range(508)]
        with patch.object(Path,'resolve',side_effect=AssertionError('no resolution')), \
             patch.object(Path,'stat',side_effect=AssertionError('no stats')):
            _,bound=self.declare(path_bindings=aliases); self.assertEqual(len(bound),512)
            aliases.append(dict(path=str(self.root/'extra-alias'),canonical_path=self.ids['source']['path']))
            with self.assertRaisesRegex(ValueError,'count exceeds bound'): self.declare(path_bindings=aliases)
            with self.assertRaisesRegex(ValueError,'conflicting'):
                self.declare(frozen_identities=self.frozen+[dict(self.ids['optional.so'],sha256='0'*64)])
            with self.assertRaisesRegex(ValueError,'unbound/conflicting'):
                self.declare(path_bindings=[dict(path=str(self.alias),canonical_path=self.ids['source']['path'])])

    def test_all_frozen_including_absent_optional_and_post_only_no_processes(self):
        before = deepcopy((self.binding,self.contract,self.frozen))
        with patch('subprocess.Popen',side_effect=AssertionError('no child processes')):
            result = self.verify()
        self.assertEqual(before,(self.binding,self.contract,self.frozen))
        rows = {Path(r['identity']['path']).name:r for r in result['files']}
        self.assertEqual(set(rows),set(self.ids))
        self.assertEqual(rows['lazy.so']['binding'],'post_only')
        self.assertFalse(rows['lazy.so']['identity']['expected_digest_verified'])
        self.assertEqual(rows['optional.so']['binding'],'prelaunch_and_post')
        self.assertFalse(rows['optional.so']['observed_executable_mapping'])
        self.assertEqual(rows['required.so']['kernel_map_device'],28)
        self.assertEqual(result['bytes_hashed'],sum(v['bytes'] for v in self.ids.values()))
        self.assertTrue(result['supplied_path_bindings_rechecked'])
        for key in ('mapped_file_bytes_verified','launch_provenance_verified','whole_run_qualified','finalization_order_verified'):
            self.assertFalse(result[key])
        result['files'][0]['identity']['path']='changed'
        self.assertEqual(before,(self.binding,self.contract,self.frozen))

    def test_unmapped_frozen_byte_mutation_and_same_byte_replacement(self):
        for name in ('source','optional.so','injection'):
            path=Path(self.ids[name]['path']); original=path.read_bytes()
            with self.subTest(name=name,mode='bytes'):
                path.write_bytes(b'X'+original[1:])
                with self.assertRaisesRegex(ValueError,'digest mismatch'): self.verify()
                path.write_bytes(original)
            # Keep old inode linked, so the filesystem cannot reuse it.
            saved=path.with_suffix('.saved'); path.rename(saved); path.write_bytes(original)
            with self.subTest(name=name,mode='replacement'), self.assertRaisesRegex(ValueError,'identity changed'):
                self.verify()
            path.unlink(); saved.rename(path)

    def test_known_observed_drift_and_injection_cannot_be_post_only(self):
        path=self.ids['required.so']['path']
        self.observed[path]['path_inode']+=1; self.observed[path]['inode']+=1
        with self.assertRaisesRegex(ValueError,'mapped pathname identity'): self.verify()
        self.observed[path]['path_inode']-=1; self.observed[path]['inode']-=1
        value=self.ids['injection']
        self.observed[value['path']]=dict(path=value['path'],device=28,inode=value['inode'],
            path_device=value['device'],path_inode=value['inode'],kernel_map_link_verified=True)
        with self.assertRaisesRegex(ValueError,'injection'): self.verify()

    def test_alias_command_and_parent_symlink_drift_before_hash(self):
        with patch.object(check,'file_identity',side_effect=AssertionError('must fail before hash')):
            self.alias.unlink(); self.alias.symlink_to('optional.so')
            with self.assertRaisesRegex(ValueError,'alias target changed'): self.verify()
            self.alias.unlink(); self.alias.symlink_to('required.so')
            self.command.unlink(); self.command.symlink_to('candidate')
            with self.assertRaisesRegex(ValueError,'alias target changed'): self.verify()
            self.command.unlink(); self.command.symlink_to('python')
            moved=self.root/'moved'; self.lib.rename(moved); self.lib.symlink_to(moved)
            with self.assertRaisesRegex(ValueError,'canonical'): self.verify()

    def test_unlisted_roots_names_non_elf_and_symlinks(self):
        with self.assertRaisesRegex(ValueError,'non-root'): self.verify(library_roots=[Path('/')])
        other=self.root/'other'; other.mkdir()
        with self.assertRaisesRegex(ValueError,'unreviewed'): self.verify(library_roots=[other])
        with self.assertRaisesRegex(ValueError,'normalized'): self.verify(library_roots=[self.lib/'..'])
        path=Path(self.ids['lazy.so']['path']); renamed=self.lib/'unreviewed-executable'
        path.rename(renamed)
        mapping=self.observed.pop(str(path)); mapping['path']=str(renamed); self.observed[str(renamed)]=mapping
        with self.assertRaisesRegex(ValueError,'unreviewed'): self.verify()

    def test_post_only_file_types_and_bounds_before_hash(self):
        path=Path(self.ids['lazy.so']['path'])
        original=path.read_bytes(); path.write_bytes(b'notELF'+original[6:])
        with self.assertRaisesRegex(ValueError,'ELF-magic'): self.verify()
        path.write_bytes(original)
        for constant, value in (('FILE_COUNT_LIMIT',1),('FILE_BYTES_LIMIT',1),('FILE_SIZE_LIMIT',1),('PATH_COUNT_LIMIT',1)):
            with self.subTest(constant=constant), patch.object(check,constant,value), \
                 patch.object(check,'file_identity',side_effect=AssertionError('no hashing over budget')):
                with self.assertRaises(ValueError): self.verify()
        saved=path.with_suffix('.saved'); path.rename(saved); os.mkfifo(path)
        with self.assertRaisesRegex(ValueError,'non-regular'): self.verify()
        path.unlink(); path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError,'canonical'): self.verify()

    def test_invalid_or_conflicting_declarations(self):
        for mode in ('known','required','frozen','alias','unbound','schema','claim'):
            contract,binding=deepcopy((self.contract,self.binding))
            frozen,aliases=deepcopy(self.frozen),[]
            if mode=='known': contract['known_libraries'].append(deepcopy(contract['known_libraries'][0]))
            if mode=='required': contract['libraries']['python']=[dict(self.ids['required.so'],sha256='0'*64)]
            if mode=='frozen': frozen.append(dict(self.ids['optional.so'],sha256='0'*64))
            if mode=='alias': aliases=[dict(path=str(self.alias),canonical_path=self.ids['optional.so']['path'])]
            if mode=='unbound': aliases=[dict(path=str(self.alias),canonical_path=str(self.root/'unknown'))]
            if mode=='schema': binding['schema_version']=True
            if mode=='claim': binding['mapped_file_bytes_verified']=True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                check.verify_files(binding,contract,frozen_identities=frozen,path_bindings=aliases,
                                   injection=self.ids['injection'],library_roots=[self.lib])

    def test_real_cpu_sequence_binding_to_file_check(self):
        if getattr(live.ClockObservationTests,'__unittest_skip__',False): self.skipTest('Linux CPU2/taskset required')
        fixture=live.ClockObservationTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        known=[*fixture.libraries['previous'],fixture.injection]
        contract=dict(schema_version=1,prepared_declaration_consistency_verified=True,
                      runtime_mapping_verified=False,launch_ready=False,workloads=fixture.workloads,
                      taskset=fixture.taskset,libraries=fixture.libraries,known_libraries=known,path_bindings=[])
        frozen=[file_identity(fixture.script)]
        roots=sorted({Path(known[0]['path']).parent, (Path(sys.base_prefix)/'lib').resolve()})
        deadline=time.monotonic()+30
        with patch.object(live.sequence,'plan_jobs',return_value=fixture.jobs):
            live.sequence.run_sequence(fixture.previous,fixture.binaries,fixture.output,root=fixture.root,
                env=fixture.env,work_deadline=deadline,cancel_check=lambda:False,
                process_observer_factory=fixture.factory(known_libraries=known),taskset=fixture.taskset['path'])
        result=json.loads((fixture.output/'sequence-result.json').read_bytes())
        binding=verify_clock_processes.validate_processes(fixture.jobs,result,contract,
                    injection=fixture.injection,work_deadline=deadline)
        checked=check.verify_files(binding,contract,frozen_identities=frozen,path_bindings=[],
                                   injection=fixture.injection,library_roots=roots)
        self.assertTrue(checked['supplied_frozen_file_bytes_rechecked'])
        self.assertTrue(any(r['binding']=='post_only' for r in checked['files']))
        self.assertTrue(all(not Path('/proc/'+str(row['pid'])).exists() for row in binding['jobs']))
        self.assertFalse(checked['whole_run_qualified'])

    def test_drift_during_hashing_including_previously_hashed_file(self):
        original=check.file_identity
        for mode in ('alias','replacement','earlier_bytes'):
            with self.subTest(mode=mode):
                changed=False
                # source sorts last; change an earlier input during its hash.
                victim=Path(self.ids['optional.so']['path']); raw=victim.read_bytes()
                saved=victim.with_suffix('.saved')
                def hashing(path,*args):
                    nonlocal changed
                    result=original(path,*args)
                    if path == Path(self.ids['source']['path']) and not changed:
                        changed=True
                        if mode=='alias':
                            self.alias.unlink(); self.alias.symlink_to('optional.so')
                        elif mode=='replacement':
                            victim.rename(saved); victim.write_bytes(raw)
                        else: victim.write_bytes(b'X'+raw[1:])
                    return result
                with patch.object(check,'file_identity',side_effect=hashing), self.assertRaises(ValueError): self.verify()
                self.assertTrue(changed)
                if mode=='alias':
                    self.alias.unlink(); self.alias.symlink_to('required.so')
                elif mode=='replacement': victim.unlink(); saved.rename(victim)
                else: victim.write_bytes(raw)


if __name__ == '__main__': unittest.main()
