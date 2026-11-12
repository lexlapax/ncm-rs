"""Entry wiring/parent/cancellation tests. Transport fixtures NEVER run GPU/model commands."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import readonly_trace_entry as entry
from bounded_child import run_bounded
from finalize_readonly_trace import sha
from readonly_trace_preflight import file_identity
from readonly_trace_telemetry import commands
from test_finalize_readonly_trace import build_fixture

HERE = Path(__file__).resolve().parent
ENV = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'LANG': 'C', 'LC_ALL': 'C'}


def transport_driver():
    """Actual -S role processes/parent guards; preflight, observation and quality are stubs.

    Only this test program patches callbacks. The production CLI has no test mode
    or validation bypass. No prepared native command or GPU query is executed.
    """
    manifest = Path(sys.argv[2])
    contract = json.loads(manifest.read_bytes())
    entry.ROOT = Path(contract['project_root'])
    entry.ENTRY = Path(__file__).resolve()
    mode = contract['_fixture_mode']
    launch = Path(sys.argv[4])
    def preflight(*args, **kwargs):
        return {'manifest': str(manifest), 'sha256': sha(manifest.read_bytes())}
    def coordinate(jobs, observer_argv, **kwargs):
        factory = kwargs['process_observer_factory']
        assert all(factory(i, job) is not None for i, job in enumerate(jobs, 1))
        if mode == 'cancel':
            escaped = subprocess.Popen([sys.executable, '-S', '-c',
                'import os,time; from pathlib import Path; '
                f'Path({str(launch / "escaped.pid")!r}).write_text(str(os.getpid())); time.sleep(30)'],
                env=kwargs['env'], start_new_session=True)
            # The surviving independent supervisor must adopt/reap this escaped
            # child after cancellation kills this collection worker.
            time.sleep(30)
            raise AssertionError('cancellation did not arrive')
        receipt = run_bounded(observer_argv, cwd=entry.ROOT, env=kwargs['env'],
                              capture_path=launch / 'test-observer.txt', receipt_path=launch / 'test-observer.receipt.json',
                              timeout_seconds=5, output_limit_bytes=1024**2)
        return {'status': receipt['status']}
    def collect(cpu, gpu, **kwargs):
        assert cpu == contract['telemetry_commands']['cpu'] and gpu == contract['telemetry_commands']['gpu']
        (launch / 'telemetry-role-called').write_text('synthetic transport only; no commands executed')
        return {'status': 'complete'}
    def finalize(raw, digest, directory, runner):
        (directory / 'preparation.json').write_bytes(raw)
        receipt = runner()
        if receipt['status'] != 'complete':
            raise ValueError('transport fixture supervisor did not complete')
        output = directory / 'completion'
        output.mkdir()
        (output / 'completion.json').write_text('{"synthetic_transport_only":true}')
        return {'status': 'failed' if mode == 'incomplete' else 'complete',
                'validation_composition_complete': mode != 'incomplete',
                'frozen_rehash_after_runner_and_validation': mode != 'incomplete',
                'performance_acceptance_evaluated': False}
    entry.prepare_contract = preflight
    entry.coordinate = coordinate
    entry.collect = collect
    entry.run_then_finalize = finalize
    raise SystemExit(entry.main())


def transport_case(mode):
    fixture, contract, _raw, launch, _collection = build_fixture()
    # build_fixture creates launch for the finalizer tests; the entry owns it here.
    launch.rmdir()
    python_id = file_identity(Path(sys.executable).resolve())
    taskset_id = file_identity(Path('/usr/bin/taskset').resolve())
    contract['utilities']['python'] = python_id
    contract['libraries']['targets']['taskset'] = taskset_id
    for job in contract['prepared']['jobs']:
        job['argv'][0] = taskset_id['path']
    contract['prepared']['file_identities'] = [v for v in contract['prepared']['file_identities']
                                               if v['path'] not in (python_id['path'], taskset_id['path'])]
    contract['prepared']['file_identities'] += [python_id, taskset_id]
    driver_id = file_identity(Path(__file__).resolve())
    contract['tool_identities'].append(driver_id)
    contract['prepared']['file_identities'].append(driver_id)
    contract['libraries']['listings']['injection'] = copy.deepcopy(contract['libraries']['listings']['accepted'])
    contract['libraries']['environment'] = contract['prepared']['observer_env']
    contract['worktrees'].update(git_identity=contract['utilities']['git'], production_base='a'*40, manifest_base='b'*40)
    contract['worktrees']['states']['candidate']['tracked_clean'] = True
    contract['worktrees']['states']['accepted'] = {'head': 'c'*40, 'tracked_clean': True}
    contract['timezone'].update(localtime_path='/synthetic/localtime', link_target='UTC')
    cpu, gpu = commands(fixture.root, Path(contract['collection_root']) / 'telemetry', python_id['path'],
                        taskset_id['path'], contract['utilities']['nvidia_smi']['path'], contract['gpu']['uuid'])
    cpu.insert(cpu.index(python_id['path']) + 1, '-S')
    contract['telemetry_commands'] = {'cpu': cpu, 'gpu': gpu}
    contract['gpu'].update(pci_bus_id='00000000:01:00.0', name='synthetic device, never queried')
    contract['_fixture_mode'] = mode
    manifest = fixture.root / 'reviewed.json'
    raw = json.dumps(contract).encode()
    manifest.write_bytes(raw)
    argv = [python_id['path'], '-S', str(Path(__file__).resolve()), 'launch', str(manifest), sha(raw),
            str(launch), '--quiet-window-confirmed']
    child = subprocess.Popen(argv, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        escaped_pid = None
        if mode == 'cancel':
            deadline = time.monotonic() + 10
            while not (launch / 'escaped.pid').exists():
                assert child.poll() is None, child.communicate()
                assert time.monotonic() < deadline, 'fixture did not start'
                time.sleep(.02)
            escaped_pid = int((launch / 'escaped.pid').read_text())
            child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=20)
        if mode == 'complete':
            assert child.returncode == 0, stdout + stderr
        supervisor = json.loads((launch / 'supervision/supervisor.json').read_bytes())
        receipt = json.loads((launch / 'supervisor.receipt.json').read_bytes())
        if mode == 'complete':
            assert child.returncode == 0, stdout + stderr
            assert supervisor['status'] == receipt['status'] == 'complete', supervisor
            assert (launch / 'telemetry-role-called').exists()
            assert (launch / 'entry-result.json').exists()
            assert receipt['termination_grace_seconds'] == 10
            assert receipt['termination_signal'] == 'not_attempted'
        elif mode == 'incomplete':
            assert child.returncode != 0 and 'finalization did not complete' in stderr
            assert supervisor['status'] == receipt['status'] == 'complete'
            assert not (launch / 'entry-result.json').exists()
        else:
            assert child.returncode != 0
            assert supervisor['status'] == receipt['status'] == 'failed'
            assert receipt['termination_signal'] == 'sent' and not receipt['termination_grace_exhausted'], receipt
            assert supervisor['cleanup']['complete'] and supervisor['subreaper_restored'], supervisor
            assert supervisor['orphan_observed'], supervisor
            assert not Path(f'/proc/{escaped_pid}').exists()
            assert not (launch / 'entry-result.json').exists()
        assert not Path(f'/proc/{receipt["pid"]}').exists()
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=15)
        fixture.doCleanups()


class EntryTests(unittest.TestCase):
    def test_no_availability_flag_means_no_preflight_or_spawn(self):
        with patch.object(entry, 'prepare_contract') as prepare, patch.object(entry, 'run_bounded') as run:
            with self.assertRaisesRegex(ValueError, 'fresh quiet-window'):
                entry.launch_collection(Path('/absent'), '0'*64, Path('/absent-launch'))
        prepare.assert_not_called()
        run.assert_not_called()

    def test_cli_help_without_execution_and_no_site_guard(self):
        run = subprocess.run([sys.executable, '-S', str(entry.ENTRY), '--help'], env=ENV,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('--quiet-window-confirmed', run.stdout)
        run = subprocess.run([sys.executable, str(entry.ENTRY), 'launch', '/absent', '0'*64, '/absent-launch'],
                             env=ENV, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('invoke entry with python -S', run.stderr)

    def test_preflight_comparison_rejects_pin_policy_or_command_changes(self):
        value = {key: {} for key in ('project_root', 'collection_root', 'expected_candidate_head', 'prepared',
                 'utilities', 'tool_identities', 'gpu', 'library_roots', 'telemetry_commands', 'limits')}
        value.update(libraries={key: {} for key in ('loader_identity', 'targets', 'libraries', 'listings',
                     'environment', 'static_loader_resolution_verified')},
                     timezone={key: 'x' for key in ('zone', 'localtime_path', 'link_target', 'file_identity')},
                     worktrees={key: 'x' for key in ('git_identity', 'production_base', 'manifest_base', 'worktree_state_verified')})
        value['worktrees']['states'] = {key: {'head': 'a'*40, 'tracked_clean': True, 'allowed_untracked_count': 1}
                                        for key in ('accepted', 'candidate')}
        entry.preflight_matches(value, copy.deepcopy(value))
        for section in ('prepared', 'utilities', 'tool_identities', 'gpu', 'telemetry_commands', 'limits', 'library_roots'):
            current = copy.deepcopy(value)
            current[section] = {'changed': True}
            with self.subTest(section=section), self.assertRaises(ValueError):
                entry.preflight_matches(value, current)
        current = copy.deepcopy(value)
        current['worktrees']['states']['candidate']['allowed_untracked_count'] = 5
        current['timezone']['observed_utc'] = 'later checkpoint'
        entry.preflight_matches(value, current)

    def test_parent_command_plan_and_image_mismatches_fail_without_serializing_values(self):
        contract = {'_manifest_sha256': 'a'*64, 'collection_root': '/synthetic/collection',
                    'libraries': {'targets': {'taskset': {'path': '/usr/bin/taskset'}}},
                    'utilities': {'python': {'path': '/usr/bin/python', 'device': 1, 'inode': 2}}}
        launch = Path('/synthetic/launch')
        plan = dict(schema_version=1, quiet_window_confirmed=True, preparation_sha256='a'*64,
                    root_pid=33, root_argv=['known-root'], worker_timeout_seconds=entry.WORKER_TIMEOUT,
                    outer_timeout_seconds=entry.OUTER_TIMEOUT, supervisor_grace_seconds=entry.SUPERVISOR_GRACE)
        raw = json.dumps(plan).encode()
        expected = b'\0'.join(v.encode() for v in entry.role_command(contract, launch, '_supervisor', 33, sha(raw))[3:]) + b'\0'
        with patch.object(entry, 'read_bounded', return_value=raw), patch.object(entry.os, 'getppid', return_value=77), \
                patch.object(entry, 'process_stat', return_value=(77, 33, 123)), \
                patch.object(entry.os, 'stat', return_value=SimpleNamespace(st_dev=1, st_ino=2)), \
                patch.object(entry, 'read_proc', return_value=expected) as read:
            entry.check_parent(contract, launch, '_coordinator', 77, sha(raw))
            with self.assertRaisesRegex(ValueError, 'plan digest mismatch'):
                entry.check_parent(contract, launch, '_coordinator', 77, '0'*64)
            read.return_value = b'unexpected-private-command\0'
            with self.assertRaises(ValueError) as caught:
                entry.check_parent(contract, launch, '_coordinator', 77, sha(raw))
            self.assertNotIn('unexpected-private-command', str(caught.exception))
            read.return_value = expected
            with patch.object(entry.os, 'stat', return_value=SimpleNamespace(st_dev=1, st_ino=3)), \
                    self.assertRaisesRegex(ValueError, 'parent image'):
                entry.check_parent(contract, launch, '_coordinator', 77, sha(raw))
            with self.assertRaisesRegex(ValueError, 'parent missing or lost'):
                entry.check_parent(contract, launch, '_coordinator', 78, sha(raw))

    def test_factory_is_mandatory_and_includes_injection_dependencies(self):
        from test_verify_readonly_processes import ProcessPostflightTests
        fixture = ProcessPostflightTests()
        fixture.setUp()
        contract = {'prepared': fixture.prepared, 'libraries': fixture.libraries}
        try:
            libraries = contract['libraries']
            libraries['listings']['injection'] = {'dependencies': {'dynamic': {'canonical_path': fixture.ids['dynamic']['path']}}}
            libraries['libraries'][fixture.ids['dynamic']['path']] = fixture.ids['dynamic']
            factory = entry.process_factory(contract)
            plain = factory(1, contract['prepared']['jobs'][0])
            traced = factory(3, contract['prepared']['jobs'][2])
            self.assertFalse(plain.traced)
            self.assertTrue(traced.traced)
            self.assertEqual(len(plain.required), 1)
            self.assertEqual(len(traced.required), 2)
            with self.assertRaises(ValueError):
                factory(2, contract['prepared']['jobs'][0])
        finally:
            fixture.doCleanups()

    @unittest.skipUnless(sys.platform == 'linux' and 0 in os.sched_getaffinity(0), 'selected Linux observer CPU0 required')
    def test_actual_internal_role_transport_without_gpu_or_benchmark_commands(self):
        # Keep archive fixture parsing out of the report-suite HWM process.
        run = subprocess.run([sys.executable, '-S', '-c',
            'from test_readonly_trace_entry import transport_case; transport_case("complete")'], cwd=HERE,
            env=ENV, capture_output=True, text=True, timeout=35)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    @unittest.skipUnless(sys.platform == 'linux' and 0 in os.sched_getaffinity(0), 'selected Linux observer CPU0 required')
    def test_root_never_publishes_entry_success_for_incomplete_finalization(self):
        run = subprocess.run([sys.executable, '-S', '-c',
            'from test_readonly_trace_entry import transport_case; transport_case("incomplete")'], cwd=HERE,
            env=ENV, capture_output=True, text=True, timeout=35)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    @unittest.skipUnless(sys.platform == 'linux' and 0 in os.sched_getaffinity(0), 'selected Linux observer CPU0 required')
    def test_outer_interrupt_allows_supervisor_to_reap_session_escape(self):
        run = subprocess.run([sys.executable, '-S', '-c',
            'from test_readonly_trace_entry import transport_case; transport_case("cancel")'], cwd=HERE,
            env=ENV, capture_output=True, text=True, timeout=35)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux waitable child cleanup')
    def test_ownership_loss_during_grace_never_sends_final_group_kill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / 'ready'
            actual_wait, actual_kill = os.waitid, os.killpg
            sent = []
            def kill(pid, sig):
                sent.append((pid, sig))
                actual_kill(pid, sig)
            def wait(*args):
                if sent:
                    os.waitpid(sent[0][0], 0)  # simulate a different reaper, only this terminated fake child
                    raise ChildProcessError('synthetic ownership loss after TERM')
                return actual_wait(*args)
            with patch('bounded_child.os.killpg', side_effect=kill), patch('bounded_child.os.waitid', side_effect=wait):
                result = run_bounded([sys.executable, '-c',
                    f'from pathlib import Path; import time; Path({str(ready)!r}).write_text("ready"); time.sleep(30)'],
                    cwd=root, env=ENV, capture_path=root / 'capture', receipt_path=root / 'receipt',
                    timeout_seconds=3, output_limit_bytes=1024, cancel_check=ready.exists, termination_grace_seconds=1)
            self.assertEqual(sent, [(result['pid'], signal.SIGTERM)])
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['direct_child_reaped'])
            self.assertIn('child_ownership_unconfirmed', result['issues'])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux waitable child cleanup')
    def test_graceful_cleanup_and_bounded_escalation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for ignore in (False, True):
                prefix = 'ignore' if ignore else 'cleanup'
                ready, cleaned = root / (prefix + '.ready'), root / (prefix + '.cleaned')
                code = ('import signal,time; from pathlib import Path; '
                        + ('signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore else
                           f'signal.signal(signal.SIGTERM,lambda *_:(Path({str(cleaned)!r}).write_text("cleaned"),exit(0))); ')
                        + f'Path({str(ready)!r}).write_text("ready"); time.sleep(30)')
                receipt = run_bounded([sys.executable, '-c', code], cwd=root, env=ENV,
                                      capture_path=root / (prefix + '.txt'), receipt_path=root / (prefix + '.json'),
                                      timeout_seconds=3, output_limit_bytes=1024, termination_grace_seconds=.2,
                                      cancel_check=ready.exists)
                self.assertEqual(receipt['status'], 'failed')
                self.assertTrue(receipt['direct_child_reaped'])
                self.assertEqual(receipt['termination_signal'], 'sent')
                self.assertEqual(receipt['termination_grace_exhausted'], ignore)
                self.assertEqual(cleaned.exists(), not ignore)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('launch', '_supervisor', '_coordinator', '_telemetry'):
        transport_driver()
    else:
        unittest.main()
