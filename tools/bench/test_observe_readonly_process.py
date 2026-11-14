"""CPU-only owned-child observation; executable mappings are test surrogates, not CUPTI."""
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bounded_child import run_bounded
import observe_readonly_process as observation
from readonly_trace_preflight import file_identity
from readonly_trace_sequence import ORDER, run_sequence

ENV = {'PATH': os.defpath, 'LANG': 'C', 'LC_ALL': 'C', 'PYTHONNOUSERSITE': '1'}


@unittest.skipUnless(sys.platform == 'linux', 'Linux owned-child proc observation')
class ProcessObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = Path(sys.executable).resolve()
        self.cpu = min(os.sched_getaffinity(0))
        self.taskset = Path('/usr/bin/taskset').resolve()
        self.injection = self.root / 'surrogate.so'
        shutil.copyfile('/usr/bin/true', self.injection)
        self.identity = file_identity(self.binary)
        self.injection_id = file_identity(self.injection)

    def spec(self, *, mapped=False, traced=False, expected_env=None, cpu=None):
        code = ('import mmap,time; '
                + (f'f=open({str(self.injection)!r},"rb"); m=mmap.mmap(f.fileno(),0,prot=mmap.PROT_READ|mmap.PROT_EXEC); '
                   if mapped else '') + 'time.sleep(.3)')
        argv = [str(self.binary), '-c', code]
        launcher = [str(self.taskset), '-c', str(self.cpu), *argv]
        observer = observation.ProcessObservation(workload=self.identity, workload_argv=argv,
                     expected_env=ENV if expected_env is None else expected_env, cpu=self.cpu if cpu is None else cpu,
                     injection=self.injection_id, traced=traced, launcher=file_identity(self.taskset),
                     launcher_argv=launcher, required_libraries=[self.identity])
        return launcher, observer

    def collect(self, argv, observer, label='child'):
        receipt = run_bounded(argv, cwd=self.root, env=ENV, capture_path=self.root / (label+'.txt'),
                              receipt_path=self.root / (label+'.receipt.json'), timeout_seconds=3,
                              output_limit_bytes=1024**2, observe_child=observer.observe)
        result = observer.finish(receipt, self.root / (label+'.process.json'))
        self.assertTrue(receipt['direct_child_reaped'])
        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())
        return receipt, result

    def test_plain_and_traced_surrogate_mapping_observation(self):
        for traced in (False, True):
            with self.subTest(traced=traced):
                argv, observer = self.spec(mapped=traced, traced=traced)
                receipt, result = self.collect(argv, observer, str(traced))
                self.assertEqual(receipt['status'], 'complete', receipt['issues'])
                self.assertEqual(result['status'], 'complete', result['issues'])
                self.assertTrue(receipt['process_observer_enabled'])
                work = [row for row in result['samples'] if row['phase'] == 'workload']
                self.assertGreaterEqual(len(work), 1)
                self.assertEqual(any(row['injection_observed'] for row in work), traced)
                self.assertTrue(all(b['monotonic_start_ns'] - a['monotonic_start_ns'] >= observation.SAMPLE_INTERVAL_NS
                                    for a, b in zip(result['samples'], result['samples'][1:])))
                for field in ('launch_provenance_verified', 'mapped_file_bytes_verified',
                              'whole_lifetime_mapping_coverage', 'environment_values_serialized'):
                    self.assertFalse(result[field])

    def test_environment_command_affinity_and_executable_mismatch_cancel_child(self):
        for mode in ('environment', 'command', 'affinity', 'executable'):
            with self.subTest(mode=mode):
                argv, observer = self.spec()
                if mode == 'environment':
                    observer.env[b'UNEXPECTED_SECRET'] = b'do-not-publish-this-value'
                elif mode == 'command':
                    observer.argv['workload'] = b'wrong\0'
                elif mode == 'affinity':
                    observer.cpu = -1
                else:
                    observer.workload = self.injection_id
                receipt, result = self.collect(argv, observer, mode)
                self.assertEqual(receipt['status'], 'failed')
                self.assertIn('capture_error:ObservationMismatch', receipt['issues'])
                self.assertEqual(result['status'], 'failed')
                self.assertNotIn('do-not-publish-this-value', json.dumps(result))
                self.assertNotIn('UNEXPECTED_SECRET', json.dumps(result))
                if mode == 'command':
                    self.assertEqual(len(result['observation_failures']), 1)
                    self.assertIn(result['observation_failures'][0], {
                        'live child command mismatch: other_declared_phase',
                        'live child command mismatch: collector_command',
                        'live child command mismatch: unclassified'})
        _, observer = self.spec()
        observer.pid = 123
        with patch.object(observer, '_observe', side_effect=observation.ObservationMismatch('live child environment mismatch')), \
                patch.object(observation.os, 'waitid', return_value=SimpleNamespace(si_pid=123)) as wait:
            with self.assertRaises(observation.ObservationMismatch):
                observer.observe(123)
        wait.assert_not_called()  # a later exit must not erase a detected mismatch

    def test_command_mismatch_categories_never_expose_command_bytes(self):
        classify=observation.command_mismatch_category
        self.assertEqual(classify(b'secret-other\0',b'secret-other\0',None),'other_declared_phase')
        self.assertEqual(classify(b'secret-parent\0',None,b'secret-parent\0'),'collector_command')
        self.assertEqual(classify(b'secret-unknown\0',b'other\0',b'parent\0'),'unclassified')
        self.assertNotIn('secret',classify(b'secret-unknown\0',None,None))

    def test_collector_command_match_is_never_accepted_as_startup(self):
        argv, observer = self.spec()
        actual = observation.read_proc
        collector = actual('/proc/self/cmdline', observation.ENV_LIMIT)
        def inherited_cmdline(path, limit):
            parts = Path(path).parts
            if len(parts) == 4 and parts[1] == 'proc' and parts[2].isdigit() and parts[3] == 'cmdline':
                return collector
            return actual(path, limit)
        with patch.object(observation, 'read_proc', side_effect=inherited_cmdline):
            receipt, result = self.collect(argv, observer, 'collector-command')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['samples'], [])
        self.assertEqual(result['observation_failures'], ['live child command mismatch: collector_command'])
        self.assertIn('workload_not_observed', result['issues'])

    def test_plain_injection_and_missing_traced_injection_fail_without_retry(self):
        argv, observer = self.spec(mapped=True, traced=False)
        receipt, result = self.collect(argv, observer, 'unexpected')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(result['status'], 'failed')
        argv, observer = self.spec(mapped=False, traced=True)
        receipt, result = self.collect(argv, observer, 'missing')
        self.assertEqual(receipt['status'], 'complete')
        self.assertIn('injection_not_observed_in_traced_workload', result['issues'])

    def test_complete_sequence_has_one_matching_observation_per_fresh_child(self):
        jobs, observers = [], []
        for condition in ORDER:
            argv, observer = self.spec(mapped=condition.endswith('T'), traced=condition.endswith('T'))
            jobs.append(dict(condition=condition, argv=argv, env=ENV, cwd=str(self.root)))
            observers.append(observer)
        with patch('readonly_trace_sequence.shutil.disk_usage', return_value=SimpleNamespace(free=100*1024**3)):
            result = run_sequence(jobs, self.root / 'sequence',
                                  process_observer_factory=lambda ordinal, job: observers[ordinal-1])
        self.assertEqual(result['status'], 'complete',
                         [(row['receipt']['issues'], observers[i].failure_reasons) for i, row in enumerate(result['jobs'])])
        pids = set()
        for ordinal, job in enumerate(result['jobs'], 1):
            path = self.root / 'sequence' / f'{ordinal:02d}-{job["condition"]}.process.json'
            observed = json.loads(path.read_text())
            self.assertEqual(observed['status'], 'complete')
            self.assertEqual(observed['pid'], job['receipt']['pid'])
            pids.add(observed['pid'])
        self.assertEqual(len(pids), 12)

    def test_incomplete_exec_startup_is_not_promoted_without_later_full_sample(self):
        actual = observation.mapped_paths
        for incomplete_forever in (False, True):
            argv, observer = self.spec()
            def pending(pid, raw):
                # Inject until the intended classification has actually occurred,
                # not merely until one attempt was lost to another startup race.
                return ([], []) if incomplete_forever or observer.startup_incomplete_count == 0 else actual(pid, raw)
            actual_stat = observation.os.stat
            image_reads = 0
            def exec_race(path, *args, **kwargs):
                nonlocal image_reads
                value = actual_stat(path, *args, **kwargs)
                if str(path).startswith('/proc/') and str(path).endswith('/exe'):
                    image_reads += 1
                    if image_reads == 2:
                        return SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino + 1,
                                               st_mode=value.st_mode)
                return value
            # The first empty-map sample can be discarded as a mixed exec before
            # startup classification. Make that ordering deterministic here.
            with patch.object(observation, 'mapped_paths', side_effect=pending), \
                    patch.object(observation.os, 'stat', side_effect=exec_race):
                receipt, result = self.collect(argv, observer, 'startup-' + str(incomplete_forever))
            self.assertGreaterEqual(image_reads, 2)
            self.assertEqual(receipt['status'], 'complete')
            self.assertGreaterEqual(result['startup_incomplete_samples'], 1)
            self.assertEqual(result['status'], 'failed' if incomplete_forever else 'complete')
            if incomplete_forever:
                self.assertIn('workload_not_observed', result['issues'])
        for error in (FileNotFoundError, ProcessLookupError, PermissionError):
            argv, observer = self.spec()
            with patch.object(observation, 'mapped_paths', side_effect=error):
                receipt, result = self.collect(argv, observer, error.__name__)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(receipt['status'], 'complete')
            if error is PermissionError:
                self.assertGreaterEqual(result['startup_permission_denied_samples'], 1)
            else:
                self.assertGreaterEqual(result['proc_unavailable_samples'], 1)
            self.assertIn('workload_not_observed', result['issues'])
        argv, observer = self.spec()
        calls = 0
        def denied_after_work(pid, raw):
            nonlocal calls
            calls += 1
            if any(row['phase'] == 'workload' for row in observer.samples):
                raise PermissionError
            return actual(pid, raw)
        with patch.object(observation, 'mapped_paths', side_effect=denied_after_work):
            receipt, result = self.collect(argv, observer, 'permission-after-work')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(result['status'], 'failed')

    def test_enabled_factory_cannot_silently_disable_observation(self):
        argv, _ = self.spec()
        jobs = [dict(condition=c, argv=argv, env=ENV, cwd=str(self.root)) for c in ORDER]
        with patch('readonly_trace_sequence.shutil.disk_usage', return_value=SimpleNamespace(free=100*1024**3)), \
                patch('readonly_trace_sequence.run_bounded', return_value={'status': 'complete', 'captured_bytes': 0}) as child:
            with self.assertRaises(ValueError):
                run_sequence(jobs, self.root / 'sequence', process_observer_factory=lambda ordinal, job: None)
        child.assert_not_called()

    def test_sequence_stops_when_observation_missing_even_if_child_succeeds(self):
        argv, observer = self.spec(traced=True)
        jobs = [dict(condition=c, argv=argv, env=ENV, cwd=str(self.root)) for c in ORDER]
        calls = []
        def factory(ordinal, job):
            calls.append(ordinal)
            return observer
        with patch('readonly_trace_sequence.shutil.disk_usage', return_value=SimpleNamespace(free=100*1024**3)):
            result = run_sequence(jobs, self.root / 'sequence', process_observer_factory=factory)
        self.assertEqual(result['stop_reason'], 'process_observation_failed')
        self.assertEqual(result['jobs'][0]['receipt']['status'], 'complete')
        self.assertEqual(calls, [1])
        self.assertFalse((self.root / 'sequence/02-CP.capture').exists())
        self.assertTrue((self.root / 'sequence/01-AP.process.json').is_file())

    def test_parser_rejects_ambiguous_deleted_and_malformed_maps(self):
        raw = b'1000-2000 r-xp 0000 08:01 123 /lib/a.so\n2000-3000 r--p 1000 08:01 123 /lib/a.so\n3000-4000 r-xp 0000 00:00 0 [vdso]\n'
        files, special = observation.executable_maps(raw)
        self.assertEqual(files, [dict(path='/lib/a.so', device=os.makedev(8, 1), inode=123)])
        self.assertEqual(special, ['[vdso]'])
        for bad in (raw.replace(b'/lib/a.so', b'/lib/a.so (deleted)'), raw.replace(b'08:01', b'bad'),
                    raw+b'4000-5000 r-xp 0000 08:01 456 /lib/a.so\n', b'broken\n'):
            with self.assertRaises(ValueError):
                observation.executable_maps(bad)
        self.assertEqual(observation.environment(b'A=one=two\0'), {b'A': b'one=two'})
        for raw in (b'A=value', b'A=1\0A=2\0'):
            with self.assertRaises(ValueError):
                observation.environment(raw)
        stat = b'123 (name with ) parens) S 42 ' + b'0 '*17 + b'456 0\n'
        self.assertEqual(observation.process_stat(stat), (123, 42, 456))

    def test_kernel_map_device_and_path_device_remain_distinct_and_permissions_fail(self):
        raw = b'1000-2000 r-xp 0000 00:1c 123 /lib/a.so\n'
        current = SimpleNamespace(st_dev=29, st_ino=123, st_mode=stat.S_IFREG | 0o755)
        with patch.object(observation.os, 'readlink', return_value='/lib/a.so'), \
                patch.object(observation.os, 'stat', return_value=current):
            files, _ = observation.mapped_paths(123, raw)
        self.assertEqual(files[0]['device'], 28)
        self.assertEqual(files[0]['path_device'], 29)
        self.assertEqual(observation.identity_key(files[0]), (29, 123))
        with patch.object(observation.os, 'readlink', side_effect=PermissionError), self.assertRaises(PermissionError):
            observation.mapped_paths(123, raw)
        current.st_ino = 456
        with patch.object(observation.os, 'readlink', return_value='/lib/a.so'), \
                patch.object(observation.os, 'stat', return_value=current), self.assertRaises(ValueError):
            observation.mapped_paths(123, raw)

    def test_unowned_pid_and_read_or_inventory_limits_fail_closed(self):
        _, observer = self.spec()
        with self.assertRaises(ChildProcessError):
            observer.observe(os.getpid())
        path = self.root / 'large'
        path.write_bytes(b'12345')
        with self.assertRaises(ValueError):
            observation.read_proc(path, 4)
        for name, value in (('MAX_SAMPLES', 0), ('MAP_INVENTORY_BYTES', 0)):
            argv, observer = self.spec()
            with patch.object(observation, name, value):
                receipt, result = self.collect(argv, observer, name)
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(result['status'], 'failed')

    def test_wrong_injection_inode_and_absent_required_library_are_not_promoted(self):
        declared = observation.ProcessObservation(workload=self.identity, workload_argv=[str(self.binary)],
                    expected_env=ENV, cpu=self.cpu, injection=self.injection_id, traced=False,
                    required_libraries=iter([self.injection_id]))
        self.assertEqual(declared.required, {observation.identity_key(self.injection_id)})
        argv, observer = self.spec(mapped=True, traced=True)
        observer.injection['inode'] += 1
        receipt, result = self.collect(argv, observer, 'inode')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(result['status'], 'failed')
        argv, observer = self.spec()
        observer.required.add((0, 12345))
        receipt, result = self.collect(argv, observer, 'library')
        self.assertEqual(receipt['status'], 'complete')
        self.assertIn('required_libraries_not_observed', result['issues'])


if __name__ == '__main__':
    unittest.main()
