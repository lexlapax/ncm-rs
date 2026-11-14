"""Keep Linux-only exclusions from suppressing unrelated report tests."""
import unittest

import run_macos_report_tests as runner


class FakeCase:
    def __init__(self, name):
        self.name = name

    def id(self):
        return self.name


class SelectionTests(unittest.TestCase):
    def test_excludes_only_listed_linux_modules(self):
        cases = [FakeCase("test_clock_role_binding.Case.test_linux"),
                 FakeCase("test_phase2_acceptance.Case.test_portable"),
                 FakeCase("test_new_diagnostic.Case.test_not_silently_excluded")]
        self.assertEqual(runner.select(cases), cases[1:])
