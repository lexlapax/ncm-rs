"""Verify report-test scheduling preserves the discovered inventory."""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import run_linux_report_tests as scheduler
from run_linux_report_tests import OVERLAPPED, partition, run_one


class Stub:
    def __init__(self, name):
        self.name = name

    def id(self):
        return self.name


class LinuxReportSchedulingTests(unittest.TestCase):
    def test_partition_preserves_every_discovered_case_once(self):
        tests = [Stub(name) for name in sorted(OVERLAPPED)] + [Stub('other.one'), Stub('other.two')]
        serial, parallel = partition(tests)
        self.assertEqual([x.id() for x in serial], ['other.one', 'other.two'])
        self.assertEqual({x.id() for x in parallel}, OVERLAPPED)
        self.assertEqual(len(serial) + len(parallel), len(tests))

    def test_isolated_failure_is_reported(self):
        class ExpectedFailure(unittest.TestCase):
            def runTest(self):
                self.fail('sentinel failure')

        name, result, output = run_one(ExpectedFailure())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.executed, [name])
        self.assertFalse(result.wasSuccessful())
        self.assertIn('sentinel failure', output)

    def test_parallel_failure_makes_full_run_fail(self):
        class Case(unittest.TestCase):
            def __init__(self, name, fail):
                super().__init__('runTest')
                self.name, self.fail_requested = name, fail

            def id(self):
                return self.name

            def runTest(self):
                if self.fail_requested:
                    self.fail('isolated failure')

        isolated = sorted(OVERLAPPED)
        suite = unittest.TestSuite([Case('serial.case', False)] + [Case(name, index == 0) for index, name in enumerate(isolated)])
        with patch.object(scheduler.sys, 'platform', 'linux'), patch.object(scheduler.unittest.defaultTestLoader, 'discover', return_value=suite), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                scheduler.main()
        self.assertEqual(error.exception.code, 1)

    def test_missing_or_duplicate_isolated_case_fails_closed(self):
        tests = [Stub(name) for name in sorted(OVERLAPPED)]
        for bad in (tests[:-1], tests + tests[:1]):
            with self.subTest(count=len(bad)), self.assertRaises(ValueError):
                partition(bad)


if __name__ == '__main__':
    unittest.main()
