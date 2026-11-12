"""Command-only regression against the existing CLI; all subprocesses are fake."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_phase2_linux_comparison as runner

ORDERS = [('previous', 'python', 'candidate'), ('python', 'candidate', 'previous'),
          ('candidate', 'previous', 'python')]
WORKLOADS = ['codec', 'bert', 'gpt2', 'qwen3', 'tinystories']


class ComparisonPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = self.root / 'accepted-checkout'
        self.old_bins = self.root / 'accepted-bin'
        self.output = self.root / 'output'
        self.previous.mkdir()
        for directory in (self.old_bins, self.root / 'target/release/examples'):
            directory.mkdir(parents=True)
            for name in ('codec_bench', 'asset_bench'):
                (directory / name).write_bytes(b'fake non-executable fixture')

    def cli(self, *, repeats=6, workloads=('codec',), cpu=2, identical=False):
        calls, monitors = [], []

        def run(argv, *, cwd, env, check):
            self.assertTrue(check)
            self.assertEqual(env, {'PATH': os.defpath, 'LANG': 'C', 'TOKENIZERS_PARALLELISM': 'false'})
            calls.append({'argv': list(argv), 'cwd': str(cwd), 'report': argv[-1]})
            Path(argv[-1]).write_text(json.dumps({'fake_command': argv}) + '\n')
            return subprocess.CompletedProcess(argv, 0)

        def popen(argv, **kwargs):
            self.assertEqual(argv[0], 'nvidia-smi')
            self.assertIn('stdout', kwargs)
            monitor = Mock()
            monitor.poll.return_value = None
            monitor.wait.return_value = 0
            monitors.append(monitor)
            return monitor

        argv = ['runner', str(self.previous), str(self.old_bins), str(self.output),
                '--repeats', str(repeats), '--workloads', *workloads]
        if cpu is not None:
            argv += ['--cpu', str(cpu), '--observer-cpu', '0']
        if identical:
            argv.append('--identical-tokenizer-binary')
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'ROOT', self.root))
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.dict(os.environ, {'PATH': os.defpath, 'LANG': 'C'}, clear=True))
            stack.enter_context(patch.object(runner, 'capture', return_value=''))
            stack.enter_context(patch.object(runner, 'validate_observer'))
            stack.enter_context(patch.object(runner.platform, 'system', return_value='Linux'))
            stack.enter_context(patch.object(runner.os, 'sched_getaffinity', return_value={0, 2, 3}, create=True))
            stack.enter_context(patch.object(runner.os, 'sched_setaffinity', create=True))
            stack.enter_context(patch.object(runner.subprocess, 'run', side_effect=run))
            stack.enter_context(patch.object(runner.subprocess, 'Popen', side_effect=popen))
            stack.enter_context(patch('builtins.print'))
            runner.main()
        self.assertEqual(len(monitors), 2)
        for monitor in monitors:
            monitor.terminate.assert_called_once_with()
            monitor.wait.assert_called_once_with()
        environment = json.loads((self.output / 'environment.json').read_bytes())
        self.assertEqual([c['argv'] for c in environment['commands']], [c['argv'] for c in calls])
        self.assertTrue(all('finished_utc' in c and 'report_sha256' in c for c in environment['commands']))
        self.assert_commands(calls, repeats, workloads, cpu, identical)
        return calls

    def assert_commands(self, calls, repeats, workloads, cpu, identical):
        # Independent expected shapes: do not derive this oracle from plan_jobs/order.
        index = 0
        for repeat in range(1, repeats + 1):
            for profile in workloads:
                for runtime in ORDERS[(repeat - 1) % 3]:
                    call = calls[index]
                    index += 1
                    cwd = self.previous if runtime == 'previous' else self.root
                    kind = 'python' if runtime == 'python' else 'rust'
                    prefix = 'codec' if profile == 'codec' else 'tokenizer-' + profile
                    report = self.output / runtime / f'{prefix}-{kind}-{repeat}.json'
                    if runtime == 'python':
                        venv = 'venv' if profile == 'codec' else 'tokenizers-venv'
                        script = 'codec_bench.py' if profile == 'codec' else 'text_tokenizer_bench.py'
                        assets = 'canonical/sudoku-easy-qualified' if profile == 'codec' else 'upstream/tokenizers-v2'
                        expected = [str(self.root / '.reference' / venv / 'bin/python'),
                                    'tools/reference/' + script, str(self.root / 'artifacts' / assets),
                                    str(self.root / '.reference/upstream-ncm')]
                    else:
                        bins = self.old_bins if runtime == 'previous' and not identical else self.root / 'target/release/examples'
                        binary = 'codec_bench' if profile == 'codec' else 'asset_bench'
                        assets = 'sudoku-easy-qualified' if profile == 'codec' else 'tokenizers-qualified'
                        expected = [str(bins / binary), str(self.root / 'artifacts/canonical' / assets)]
                    expected += (['tests/fixtures/codec-bench-v1.json'] if profile == 'codec' else
                                 ['tests/fixtures/tokenizers-v1.json', profile])
                    expected.append(str(report))
                    if cpu is not None:
                        expected = ['taskset', '-c', str(cpu)] + expected
                    self.assertEqual(call, {'argv': expected, 'cwd': str(cwd), 'report': str(report)})
                    if runtime == 'python' and report.exists():
                        for destination in ('previous', 'candidate'):
                            self.assertEqual((self.output / destination / report.name).read_bytes(), report.read_bytes())
        self.assertEqual(index, len(calls))

    def test_existing_cli_codec_eighteen_commands_and_python_copies(self):
        self.cli()

    def test_existing_cli_all_workloads_and_unpinned_commands(self):
        self.cli(repeats=3, workloads=WORKLOADS, cpu=None)

    def test_existing_cli_identical_tokenizer_control_keeps_candidate_binary(self):
        self.cli(repeats=3, workloads=['gpt2', 'tinystories'], identical=True)

    def plan(self, **changes):
        options = dict(root=self.root, repeats=6, workloads=('codec',), cpu=2)
        return runner.plan_jobs(self.previous, self.old_bins, self.output, **(options | changes))

    def test_plan_is_side_effect_free_and_matches_existing_cli(self):
        with ExitStack() as stack:
            for method in ('open', 'read_bytes', 'read_text', 'resolve', 'exists', 'mkdir'):
                stack.enter_context(patch.object(Path, method, side_effect=AssertionError('planner filesystem IO')))
            for target in ('capture', 'subprocess.run', 'subprocess.Popen', 'os.sched_getaffinity'):
                stack.enter_context(patch('run_phase2_linux_comparison.' + target,
                                          side_effect=AssertionError('planner probe/spawn'), create=True))
            jobs = self.plan()
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({job['report'] for job in jobs}), 18)
        calls = self.cli()
        self.assertEqual([{key: job[key] for key in ('argv', 'cwd', 'report')} for job in jobs], calls)
        for job in jobs:
            expected = ([str(self.output / destination / Path(job['report']).name)
                         for destination in ('previous', 'candidate')] if job['runtime'] == 'python' else [])
            self.assertEqual(job['python_copies'], expected)
            self.assertEqual(job['workload'], 'codec')
            self.assertIn(job['repeat'], range(1, 7))
        self.assertEqual(json.loads(json.dumps(jobs)), jobs)

    def test_explicit_taskset_and_fresh_plans_do_not_share_mutable_commands(self):
        first = self.plan(taskset='/usr/bin/taskset')
        self.assertTrue(all(job['argv'][:3] == ['/usr/bin/taskset', '-c', '2'] for job in first))
        first[0]['argv'].append('mutation')
        first[1]['python_copies'].clear()
        second = self.plan(taskset='/usr/bin/taskset')
        self.assertNotIn('mutation', second[0]['argv'])
        self.assertEqual(len(second[1]['python_copies']), 2)

    def test_invalid_plan_options_fail_without_outputs(self):
        for options in ({'repeats': 2}, {'repeats': True}, {'workloads': []},
                        {'workloads': ['codec', 'codec']}, {'workloads': ['unknown']},
                        {'cpu': -1}, {'cpu': True}, {'taskset': ''}, {'taskset': 'bad\0argv'},
                        {'root': Path('relative')}, {'identical_tokenizer_binary': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
