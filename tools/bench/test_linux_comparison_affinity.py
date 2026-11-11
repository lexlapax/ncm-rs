"""Observer placement tests; no GPU, benchmark, or global scheduling changes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_phase2_linux_comparison import observer_affinity, validate_observer
from sample_linux_cpu import cpu_list


class ComparisonAffinityTests(unittest.TestCase):
    def topology(self, root, text='2,3\n'):
        path = root/'cpu2/topology/thread_siblings_list'
        path.parent.mkdir(parents=True)
        path.write_text(text)

    @patch('run_phase2_linux_comparison.os.sched_getaffinity', return_value={0, 2, 3}, create=True)
    def test_valid_placement_and_rejections(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.topology(root)
            validate_observer(2, 0, root)
            for cpu, observer in ((None, 0), (2, 2), (2, 3), (2, 4), (4, 0)):
                with self.subTest(cpu=cpu, observer=observer), self.assertRaises(ValueError):
                    validate_observer(cpu, observer, root)

    @patch('run_phase2_linux_comparison.os.sched_getaffinity', return_value={0, 2, 3}, create=True)
    def test_missing_or_inconsistent_topology_fails_closed(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                validate_observer(2, 0, root)
            self.topology(root, '0,3\n')
            with self.assertRaises(ValueError):
                validate_observer(2, 0, root)

    @patch('run_phase2_linux_comparison.os.sched_setaffinity', create=True)
    @patch('run_phase2_linux_comparison.os.sched_getaffinity', create=True)
    def test_default_does_not_touch_affinity_or_topology(self, get_affinity, set_affinity):
        validate_observer(None, None, Path('/nonexistent'))
        with observer_affinity(None):
            pass
        get_affinity.assert_not_called()
        set_affinity.assert_not_called()

    @patch('run_phase2_linux_comparison.os.sched_getaffinity', return_value={0, 2, 3}, create=True)
    @patch('run_phase2_linux_comparison.os.sched_setaffinity', create=True)
    def test_restores_on_success_and_failure(self, set_affinity, _):
        for fail in (False, True):
            set_affinity.reset_mock()
            try:
                with observer_affinity(0):
                    if fail:
                        raise RuntimeError('child failed')
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual([call.args for call in set_affinity.call_args_list],
                             [(0, {0}), (0, {0, 2, 3})])

    @patch('run_phase2_linux_comparison.os.sched_getaffinity', return_value={0, 2}, create=True)
    @patch('run_phase2_linux_comparison.os.sched_setaffinity', side_effect=OSError('denied'), create=True)
    def test_placement_failure_does_not_start_children(self, set_affinity, _):
        entered = False
        with self.assertRaises(OSError):
            with observer_affinity(0):
                entered = True
        self.assertFalse(entered)
        set_affinity.assert_called_once_with(0, {0})

    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset'), 'Linux taskset required')
    def test_real_child_inheritance_and_workload_override(self):
        original = os.sched_getaffinity(0)
        target = min(original)
        siblings = cpu_list(Path(f'/sys/devices/system/cpu/cpu{target}/topology/thread_siblings_list').read_text())
        observers = original - set(siblings)
        if not observers:
            self.skipTest('no permitted CPU outside workload SMT siblings')
        observer = min(observers)
        validate_observer(target, observer)
        command = [sys.executable, '-c', 'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))']
        with observer_affinity(observer):
            self.assertEqual(json.loads(subprocess.check_output(command)), [observer])
            self.assertEqual(json.loads(subprocess.check_output(
                ['taskset', '-c', str(target)] + command)), [target])
            self.assertEqual(os.sched_getaffinity(0), {observer})
        self.assertEqual(os.sched_getaffinity(0), original)


if __name__ == '__main__':
    unittest.main()
