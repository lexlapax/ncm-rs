"""Synthetic records/ELF-magic files plus a Linux writer-reader test; no model/GPU execution."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from readonly_trace_preflight import file_identity
from readonly_trace_sequence import ORDER
import verify_readonly_processes as postflight


class ProcessPostflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.libroot = self.root / 'lib'
        self.libroot.mkdir()
        self.ids = {}
        for name in ('accepted', 'candidate', 'injection', 'taskset', 'loader', 'native', 'dynamic'):
            path = self.libroot / (name + '.so') if name in ('native', 'dynamic') else self.root / name
            path.write_bytes(b'\x7fELF' + bytes(60) + name.encode())
            self.ids[name] = file_identity(path)
        source = self.root / 'source.rs'
        source.write_text('retained source input\n')
        self.ids['source'] = file_identity(source)
        for name in ('archive-declaration', 'profile-declaration'):
            path = self.root / (name + '.json')
            path.write_text('{}')
            self.ids[name] = file_identity(path)
        env = {'PATH': os.defpath, 'HOME': str(self.root), 'LANG': 'C', 'LC_ALL': 'C', 'PYTHONNOUSERSITE': '1'}
        self.prepared = {'jobs': [], 'observer_env': env, 'workload_cpu': 2, 'observer_cpu': 0,
                         'declaration_identities': [self.ids['archive-declaration'], self.ids['profile-declaration']],
                         'file_identities': [self.ids[name] for name in ('accepted', 'candidate', 'injection', 'taskset', 'source')]}
        dependencies = {'native': {'canonical_path': self.ids['native']['path']}}
        self.libraries = {'static_loader_resolution_verified': True,
                          'targets': {name: self.ids[name] for name in ('accepted', 'candidate', 'injection', 'taskset')},
                          'loader_identity': self.ids['loader'],
                          'libraries': {self.ids['native']['path']: self.ids['native']},
                          'listings': {name: {'dependencies': copy.deepcopy(dependencies)} for name in ('accepted', 'candidate', 'injection')}}
        self.jobs, self.observations = [], []
        for ordinal, condition in enumerate(ORDER, 1):
            runtime = 'accepted' if condition[0] == 'A' else 'candidate'
            report = str(self.root / f'{ordinal:02d}-{condition}.report.json')
            argv = [self.ids['taskset']['path'], '-c', '2', self.ids[runtime]['path'], '/artifact', '/fixture', report]
            child_env = dict(env)
            if condition.endswith('T'):
                child_env['CUDA_INJECTION64_PATH'] = self.ids['injection']['path']
            self.prepared['jobs'].append(dict(condition=condition, argv=argv, cwd=str(self.root), env=child_env, report_path=report))
            receipt = dict(status='complete', issues=[], returncode=0, direct_child_reaped=True, capture_eof=True,
                           interrupt_signal=None, process_observer_enabled=True, pid=100+ordinal, argv=argv, cwd=str(self.root))
            self.jobs.append(dict(ordinal=ordinal, condition=condition, receipt=receipt))
            names = [runtime, 'native', 'dynamic'] + (['injection'] if condition.endswith('T') else [])
            mapping = {'files': sorted([self.mapping(name) for name in names], key=lambda v: v['path']),
                       'non_file_executable_mappings': ['[vdso]']}
            digest = postflight.digest(mapping)
            sample = dict(monotonic_start_ns=ordinal*10**9, monotonic_end_ns=ordinal*10**9+1000,
                          phase='workload', affinity=[2], mapping_sha256=digest, injection_observed=condition.endswith('T'))
            self.observations.append(dict(schema_version=1, status='complete', issues=[], observation_failures=[],
                pid=100+ordinal, process_start_ticks=123, samples=[sample], mapping_sets={digest: mapping},
                transient_samples_discarded=0, startup_incomplete_samples=0, proc_unavailable_samples=0,
                startup_permission_denied_samples=0, launch_provenance_verified=False,
                whole_lifetime_mapping_coverage=False, mapped_file_bytes_verified=False, environment_values_serialized=False))

    def mapping(self, name):
        value = self.ids[name]
        return dict(path=value['path'], device=28, inode=value['inode'], path_device=value['device'],
                    path_inode=value['inode'], kernel_map_link_verified=True)

    def bind(self, *, jobs=None, observations=None, prepared=None):
        return postflight.validate_processes(self.jobs if jobs is None else jobs,
                 self.observations if observations is None else observations,
                 self.prepared if prepared is None else prepared, self.libraries)

    def files(self, binding=None):
        return postflight.verify_files(self.bind() if binding is None else binding,
                                      self.prepared, self.libraries, library_roots=[self.libroot])

    def mutate_map(self, observations, index, mutate):
        record = observations[index]
        mapping = copy.deepcopy(next(iter(record['mapping_sets'].values())))
        mutate(mapping)
        digest = postflight.digest(mapping)
        record['mapping_sets'] = {digest: mapping}
        for sample in record['samples']:
            sample['mapping_sha256'] = digest

    def test_all_processes_and_prelaunch_inputs_with_explicit_post_only_library(self):
        binding = self.bind()
        result = self.files(binding)
        self.assertEqual(len(binding['jobs']), 12)
        self.assertTrue(result['prelaunch_file_bytes_rechecked'])
        self.assertTrue(result['current_mapped_paths_verified'])
        self.assertFalse(result['mapped_file_bytes_verified'])
        self.assertFalse(result['launch_provenance_verified'])
        by_path = {row['identity']['path']: row for row in result['files']}
        self.assertEqual(len(by_path), 10)
        self.assertEqual(by_path[self.ids['dynamic']['path']]['binding'], 'post_only')
        self.assertFalse(by_path[self.ids['dynamic']['path']]['identity']['expected_digest_verified'])
        self.assertEqual(by_path[self.ids['native']['path']]['kernel_map_device'], 28)
        self.assertFalse(by_path[self.ids['source']['path']]['observed_executable_mapping'])
        self.assertEqual(result['bytes_hashed'], sum(v['bytes'] for v in self.ids.values()))
        json.dumps(result, allow_nan=False)

    def test_traced_injection_dependencies_must_be_observed(self):
        extra = self.ids['loader']  # Not mapped by the synthetic workloads.
        self.libraries['libraries'][extra['path']] = extra
        self.libraries['listings']['injection']['dependencies']['extra'] = {'canonical_path': extra['path']}
        with self.assertRaisesRegex(ValueError, 'libraries not observed'):
            self.bind()

    def test_pid_order_flags_receipts_and_declared_environment_failures(self):
        for mutate in (lambda j: j.pop(), lambda j: j.reverse(),
                       lambda j: j[0]['receipt'].update(process_observer_enabled=False),
                       lambda j: j[0]['receipt'].update(returncode=True),
                       lambda j: j[0]['receipt'].update(status='failed'),
                       lambda j: j[1]['receipt'].update(pid=101),
                       lambda j: j[0]['receipt'].update(argv=['wrong'])):
            jobs = copy.deepcopy(self.jobs)
            mutate(jobs)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.bind(jobs=jobs)
        prepared = copy.deepcopy(self.prepared)
        prepared['jobs'][0]['env']['CUDA_INJECTION64_PATH'] = self.ids['injection']['path']
        with self.assertRaisesRegex(ValueError, 'command/environment'):
            self.bind(prepared=prepared)
        prepared = copy.deepcopy(self.prepared)
        prepared['observer_cpu'] = False
        with self.assertRaisesRegex(ValueError, 'CPUs'):
            self.bind(prepared=prepared)

    def test_corrupt_mapping_digest_keys_inode_and_cross_process_change(self):
        records = copy.deepcopy(self.observations)
        next(iter(records[0]['mapping_sets'].values()))['files'][0]['inode'] += 1
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.bind(observations=records)
        for mutate in (lambda m: m['files'][0].update(kernel_map_link_verified=False),
                       lambda m: m['files'][0].update(path_inode=123),
                       lambda m: m['files'].append(m['files'][0]),
                       lambda m: m['files'][0].update(extra='unsupported')):
            records = copy.deepcopy(self.observations)
            self.mutate_map(records, 0, mutate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.bind(observations=records)
        records = copy.deepcopy(self.observations)
        self.mutate_map(records, 1, lambda m: next(v for v in m['files'] if v['path']==self.ids['native']['path']).update(device=27))
        with self.assertRaisesRegex(ValueError, 'across processes'):
            self.bind(observations=records)

    def test_scope_claims_counters_timing_affinity_and_missing_required_work(self):
        for mutate in (lambda r: r.update(mapped_file_bytes_verified=True),
                       lambda r: r.update(startup_permission_denied_samples=1),
                       lambda r: r.update(observation_failures=['failure']),
                       lambda r: r['samples'][0].update(affinity=[0]),
                       lambda r: r['samples'][0].update(monotonic_end_ns=0),
                       lambda r: r['samples'].append(dict(r['samples'][0])),
                       lambda r: r['samples'][0].update(injection_observed=True)):
            records = copy.deepcopy(self.observations)
            mutate(records[0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.bind(observations=records)
        records = copy.deepcopy(self.observations)
        self.mutate_map(records, 0, lambda m: m.update(files=[v for v in m['files'] if v['path'] != self.ids['native']['path']]))
        with self.assertRaisesRegex(ValueError, 'required native libraries'):
            self.bind(observations=records)

    def test_plain_and_traced_injection_must_match_observed_mapping_not_flag(self):
        records = copy.deepcopy(self.observations)
        self.mutate_map(records, 0, lambda m: m.update(files=sorted(m['files']+[self.mapping('injection')],key=lambda v:v['path'])))
        records[0]['samples'][0]['injection_observed'] = True
        with self.assertRaisesRegex(ValueError, 'plain'):
            self.bind(observations=records)
        records = copy.deepcopy(self.observations)
        self.mutate_map(records, 2, lambda m: m.update(files=[v for v in m['files'] if v['path'] != self.ids['injection']['path']]))
        records[2]['samples'][0]['injection_observed'] = False
        with self.assertRaisesRegex(ValueError, 'traced injection'):
            self.bind(observations=records)

    def test_changed_unmapped_input_and_mapped_file_are_not_promoted(self):
        binding = self.bind()
        source = Path(self.ids['source']['path'])
        source.write_bytes(b'X' * self.ids['source']['bytes'])  # same inode and length, different bytes
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.files(binding)
        source.write_text('retained source input\n')
        declaration = Path(self.ids['archive-declaration']['path'])
        declaration.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.files(binding)
        declaration.write_text('{}')
        native = Path(self.ids['native']['path'])
        changed = bytearray(native.read_bytes())
        changed[-1] ^= 1
        native.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.files(binding)

    def test_replacement_symlink_outside_root_and_budget_reject_before_hashing(self):
        binding = self.bind()
        dynamic = Path(self.ids['dynamic']['path'])
        replacement = self.root / 'replacement'
        shutil.copyfile(dynamic, replacement)
        dynamic.unlink()
        dynamic.symlink_to(replacement)
        with patch.object(postflight, 'file_identity') as hash_file, self.assertRaises(ValueError):
            self.files(binding)
        hash_file.assert_not_called()
        dynamic.unlink()
        os.replace(replacement, dynamic)
        with patch.object(postflight, 'file_identity') as hash_file, self.assertRaises(ValueError):
            self.files(binding)
        hash_file.assert_not_called()
        # Update only synthetic observation identity for the remaining independent guards.
        value = dynamic.stat()
        binding['observed_files'][str(dynamic)].update(inode=value.st_ino,path_inode=value.st_ino)
        (self.root / 'other').mkdir()
        with patch.object(postflight, 'file_identity') as hash_file, self.assertRaises(ValueError):
            postflight.verify_files(binding,self.prepared,self.libraries,library_roots=[self.root/'other'])
        hash_file.assert_not_called()
        for limit in ('FILE_COUNT_LIMIT', 'FILE_BYTES_LIMIT'):
            with patch.object(postflight, limit, 1), patch.object(postflight, 'file_identity') as hash_file, self.assertRaises(ValueError):
                self.files(binding)
            hash_file.assert_not_called()

    @unittest.skipUnless(sys.platform == 'linux' and Path('/usr/lib/libc.so.6').is_file()
                         and 2 in os.sched_getaffinity(0), 'native Linux fixture needs CPU2 and the local libc path')
    def test_real_observation_writer_and_postflight_reader_in_fresh_interpreter(self):
        # Synthetic loader declaration, but actual prelaunch file identities and
        # actual owned-child records. It is not a native codec or loader-probe test.
        script = '''
import json, os
from pathlib import Path
import test_observe_readonly_process as fixture
from readonly_trace_preflight import file_identity
from readonly_trace_sequence import ORDER
from verify_readonly_processes import validate_processes, verify_files
case = fixture.ProcessObservationTests(); case.setUp(); case.cpu = 2
try:
    base = dict(fixture.ENV, HOME=str(case.root))
    libc = file_identity(Path('/usr/lib/libc.so.6').resolve())
    loader = file_identity(Path('/usr/lib/ld-linux-x86-64.so.2').resolve())
    targets = dict(accepted=case.identity, candidate=case.identity, injection=case.injection_id,
                   taskset=file_identity(case.taskset))
    libraries = dict(static_loader_resolution_verified=True, targets=targets, loader_identity=loader,
                     libraries={libc['path']: libc}, listings={key: {'dependencies': {'libc': {'canonical_path': libc['path']}}}
                                                              for key in ('accepted', 'candidate', 'injection')})
    prepared = dict(observer_env=base, workload_cpu=2, observer_cpu=0, jobs=[], declaration_identities=[],
                    file_identities=[case.identity, case.injection_id, targets['taskset']])
    observers = []
    for ordinal, condition in enumerate(ORDER, 1):
        fixture.ENV = dict(base)
        if condition.endswith('T'): fixture.ENV['CUDA_INJECTION64_PATH'] = str(case.injection)
        argv, observer = case.spec(mapped=condition.endswith('T'), traced=condition.endswith('T'))
        report = str(case.root / f'{ordinal}.unused-report')
        argv.append(report)
        observer.argv['launcher'] = b'\\0'.join(a.encode() for a in argv) + b'\\0'
        observer.argv['workload'] = b'\\0'.join(a.encode() for a in argv[3:]) + b'\\0'
        observers.append(observer)
        prepared['jobs'].append(dict(condition=condition, argv=argv, cwd=str(case.root),
                                     env=dict(fixture.ENV), report_path=report))
    jobs, observations = [], []
    for ordinal, (job, observer) in enumerate(zip(prepared['jobs'], observers), 1):
        fixture.ENV = job['env']
        receipt, observed = case.collect(job['argv'], observer, str(ordinal))
        assert receipt['status'] == 'complete', (receipt['issues'], observer.failure_reasons)
        jobs.append(dict(ordinal=ordinal, condition=job['condition'], receipt=receipt))
        observations.append(observed)
    bound = validate_processes(jobs, observations, prepared, libraries)
    checked = verify_files(bound, prepared, libraries, library_roots=[Path('/usr/lib')])
    assert len(bound['jobs']) == 12 and checked['prelaunch_file_bytes_rechecked']
    assert not checked['mapped_file_bytes_verified'] and not checked['launch_provenance_verified']
    print('12 actual CPU-child records and postflight hashes checked; not codec/loader qualification')
finally:
    case.doCleanups()
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent,
                                env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'},
                                capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('12 actual CPU-child', result.stdout)

    def test_invalid_pin_or_non_elf_dynamic_file_fails(self):
        binding = self.bind()
        prepared = copy.deepcopy(self.prepared)
        prepared['file_identities'][0]['sha256'] = None
        with patch.object(postflight, 'file_identity') as hash_file, self.assertRaises(ValueError):
            postflight.verify_files(binding, prepared, self.libraries, library_roots=[self.libroot])
        hash_file.assert_not_called()
        Path(self.ids['dynamic']['path']).write_bytes(b'not an ELF')
        with self.assertRaisesRegex(ValueError, 'ELF'):
            self.files(binding)


if __name__ == '__main__':
    unittest.main()
