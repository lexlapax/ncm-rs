"""Actual CPU-only /proc observations over fixed-order fake codec jobs; no GPU use."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import clock_codec_sequence as sequence
from clock_process_observation import process_factory
from observe_readonly_process import executable_maps
from readonly_trace_preflight import file_identity
from run_phase2_linux_comparison import plan_jobs

SCRIPT = '''import json,os,sys,time
from pathlib import Path
if sys.argv[1]=='env_change' and 'UNEXPECTED' not in os.environ:
    os.execve(sys.executable,[sys.executable,*sys.orig_argv[1:]],dict(os.environ,UNEXPECTED='not-for-observation-output'))
Path(sys.argv[2]).write_text(json.dumps({'fake':True}))
time.sleep(.2)
'''


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset')
                     and 2 in os.sched_getaffinity(0), 'Linux CPU2/taskset required')
class ClockObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous, self.binaries = self.root / 'previous', self.root / 'binaries'
        self.previous.mkdir(); self.binaries.mkdir()
        self.script = self.root / 'fake.py'
        self.script.write_text(SCRIPT)
        self.python = str(Path(sys.executable).resolve())
        self.alias = self.root / 'python-alias'
        self.alias.symlink_to(self.python)
        self.taskset = file_identity(Path(shutil.which('taskset')).resolve())
        self.injection = file_identity(Path(shutil.which('true')).resolve())
        image = file_identity(self.python)
        maps, _ = executable_maps(Path('/proc/self/maps').read_bytes())
        libc = next(row['path'] for row in maps if Path(row['path']).name == 'libc.so.6')
        library = file_identity(Path(libc).resolve())
        self.workloads = {runtime: {'command_path': str(self.alias) if runtime == 'python' else self.python,
                                    'image': deepcopy(image)}
                          for runtime in ('previous', 'python', 'candidate')}
        self.libraries = {runtime: [deepcopy(library)] for runtime in self.workloads}
        self.env = {'PATH': os.defpath, 'HOME': str(self.root), 'LANG': 'C', 'LC_ALL': 'C',
                    'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        self.jobs_for('collection')

    def jobs_for(self, name, mode='ok'):
        self.output = self.root / name
        self.jobs = plan_jobs(self.previous, self.binaries, self.output, root=self.root,
                              repeats=6, workloads=('codec',), cpu=2, taskset=self.taskset['path'])
        for job in self.jobs:
            job['argv'] = [self.taskset['path'], '-c', '2', self.workloads[job['runtime']]['command_path'],
                           '-S', str(self.script), mode, job['report']]

    def factory(self, **changes):
        args = dict(env=self.env, workloads=self.workloads, taskset=self.taskset,
                    injection=self.injection, libraries=self.libraries)
        args.update(changes)
        if 'known_libraries' not in args:
            args['known_libraries'] = [v for values in args['libraries'].values() for v in values]
        return process_factory(self.jobs, **args)

    def run_sequence(self, factory):
        with patch.object(sequence, 'plan_jobs', return_value=self.jobs):
            return sequence.run_sequence(self.previous, self.binaries, self.output,
                                         root=self.root, env=self.env, work_deadline=time.monotonic() + 30,
                                         cancel_check=lambda: False, process_observer_factory=factory,
                                         taskset=self.taskset['path'])

    def test_factory_has_no_io_and_owns_independent_frozen_state(self):
        expected_jobs, expected_env = deepcopy(self.jobs), dict(self.env)
        known = deepcopy(self.libraries['previous'])
        with patch('builtins.open', side_effect=AssertionError('no IO')), \
                patch.object(Path, 'open', side_effect=AssertionError('no IO')), \
                patch.object(Path, 'resolve', side_effect=AssertionError('no resolution')), \
                patch('os.stat', side_effect=AssertionError('no metadata IO')), \
                patch('subprocess.Popen', side_effect=AssertionError('no spawn')), \
                patch('hashlib.sha256', side_effect=AssertionError('no hashing')):
            factory = self.factory(known_libraries=known)
            known[0]['inode'] += 1
            self.jobs[0]['argv'].append('changed-after-freeze')
            self.env['HOME'] = '/different-after-freeze'
            self.workloads['previous']['image']['inode'] += 1
            self.libraries['previous'][0]['inode'] += 1
            observers = [factory(i, job, expected_env) for i, job in enumerate(expected_jobs, 1)]
            self.assertEqual(len({id(value) for value in observers}), 18)
            self.assertEqual(len({id(value.samples) for value in observers}), 18)
            self.assertEqual(len({id(value.known) for value in observers}), 18)
            self.assertEqual(observers[0].known[known[0]['path']][1] + 1, known[0]['inode'])
            with self.assertRaisesRegex(ValueError, 'already issued'):
                factory(1, expected_jobs[0], expected_env)
        self.assertEqual(observers[0].workload['inode'] + 1, self.workloads['previous']['image']['inode'])
        self.assertNotIn((self.libraries['previous'][0]['device'], self.libraries['previous'][0]['inode']),
                         observers[0].required)
        self.assertEqual(observers[1].argv['workload'].split(b'\0')[0], str(self.alias).encode())
        self.assertEqual(observers[1].workload['path'], self.python)

    def test_real_observations_cover_all_eighteen_jobs_and_preserve_limits(self):
        # The unrelated true executable is a known candidate that never loads.
        known = [*self.libraries['previous'], self.injection]
        result = self.run_sequence(self.factory(known_libraries=known))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['jobs']), 18)
        for job, row in zip(self.jobs, result['jobs']):
            receipt, observation = row['receipt'], row['observation']
            self.assertTrue(receipt['direct_child_reaped'])
            self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())
            self.assertEqual(receipt['argv'], job['argv'])
            self.assertEqual(observation['status'], 'complete')
            self.assertEqual(observation['pid'], receipt['pid'])
            self.assertTrue(any(sample['phase'] == 'workload' for sample in observation['samples']))
            self.assertFalse(any(sample['injection_observed'] for sample in observation['samples']))
            for key in ('launch_provenance_verified', 'whole_lifetime_mapping_coverage',
                        'mapped_file_bytes_verified', 'environment_values_serialized'):
                self.assertFalse(observation[key])
            for copy in job['python_copies']:
                self.assertFalse(Path(copy).exists())
        self.assertFalse(result['benchmark_outputs_qualified'])
        self.assertFalse(result['whole_run_supervision_provided'])

    def test_live_mismatches_and_missing_library_stop_before_second_job(self):
        for mode in ('image', 'library', 'injection', 'env_change'):
            with self.subTest(mode=mode):
                self.jobs_for(mode, mode='env_change' if mode == 'env_change' else 'ok')
                options = {}
                if mode == 'image':
                    workloads = deepcopy(self.workloads)
                    workloads['previous']['image']['inode'] += 1
                    options['workloads'] = workloads
                elif mode == 'library':
                    libraries = deepcopy(self.libraries)
                    libraries['previous'] = [self.injection]
                    options['libraries'] = libraries
                elif mode == 'injection':
                    options['injection'] = self.libraries['previous'][0]
                result = self.run_sequence(self.factory(**options))
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(len(result['jobs']), 1)
                row = result['jobs'][0]
                self.assertTrue(row['receipt']['direct_child_reaped'])
                self.assertEqual(row['observation']['status'], 'failed')
                self.assertFalse(Path(self.jobs[1]['report']).exists())
                if mode == 'library':
                    self.assertEqual(row['receipt']['status'], 'complete')
                    self.assertIn('required_libraries_not_observed', row['observation']['issues'])
                else:
                    self.assertIn('process_observer_failed', row['observation']['issues'])
                self.assertNotIn('not-for-observation-output', json.dumps(row['observation']))

    def test_known_optional_library_mismatch_fails_first_child_and_latches(self):
        maps, _ = executable_maps(Path('/proc/self/maps').read_bytes())
        libm = next(row['path'] for row in maps if Path(row['path']).name == 'libm.so.6')
        optional = file_identity(Path(libm).resolve())
        optional['inode'] += 1
        factory = self.factory(known_libraries=[*self.libraries['previous'], optional])
        issued = []
        def capture(ordinal, job, env):
            observer = factory(ordinal, job, env)
            issued.append(observer)
            return observer
        result = self.run_sequence(capture)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['jobs']), 1)
        row = result['jobs'][0]
        self.assertTrue(row['receipt']['direct_child_reaped'])
        self.assertIn('known library identity differs from preparation', row['observation']['observation_failures'])
        self.assertIn('process_observer_failed', row['observation']['issues'])
        self.assertFalse(Path(self.jobs[1]['report']).exists())
        # Even a later absent/reaped process cannot erase the recorded mismatch.
        observer = issued[0]
        observer.known.clear()
        again = observer.finish(row['receipt'], self.root/'failure-retained.json')
        self.assertIn('process_observer_failed', again['issues'])

    def test_known_inventory_required_complete_consistent_and_bounded(self):
        known = deepcopy(self.libraries['previous'])
        changed = deepcopy(known[0]); changed['inode'] += 1
        for values in ([], [self.injection], [*known, changed], known*257,
                       [dict(known[0], path_device=known[0]['device'])],
                       [dict(known[0], path_inode=known[0]['inode'])]):
            with self.subTest(size=len(values)), patch.object(subprocess, 'Popen') as spawn:
                with self.assertRaises(ValueError): self.factory(known_libraries=values)
                spawn.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_drift_in_last_job_or_environment_rejects_before_first_spawn(self):
        for field in ('cwd', 'env', 'ordinal'):
            with self.subTest(field=field):
                self.jobs_for(field)
                factory = self.factory()
                if field == 'cwd':
                    self.jobs[-1]['cwd'] = '/changed-after-freeze'
                def callback(ordinal, job, env):
                    if ordinal == 18:
                        if field == 'env':
                            env['HOME'] = '/changed-after-freeze'
                        if field == 'ordinal':
                            ordinal = True
                    return factory(ordinal, job, env)
                with patch.object(subprocess, 'Popen') as launch:
                    with self.assertRaises(ValueError):
                        self.run_sequence(callback)
                    launch.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_incomplete_declarations_and_late_bad_plan_are_not_observers(self):
        bad = []
        workloads = deepcopy(self.workloads); del workloads['python']
        bad.append({'workloads': workloads})
        workloads = deepcopy(self.workloads); workloads['python']['image']['sha256'] = 'not-a-digest'
        bad.append({'workloads': workloads})
        libraries = deepcopy(self.libraries); libraries['candidate'] = []
        bad.append({'libraries': libraries})
        bad.append({'env': dict(self.env, LD_PRELOAD='/not-allowed')})
        libraries = deepcopy(self.libraries); libraries['previous'][0]['path_inode'] = 1
        bad.append({'libraries': libraries})
        for options in bad:
            with self.subTest(fields=list(options)), self.assertRaises(ValueError):
                self.factory(**options)
        self.jobs[-1]['argv'][2] = '0'
        with self.assertRaises(ValueError):
            self.factory()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
