"""Run portable report tests on macOS; Linux diagnostic suites run on Linux."""
import sys
import unittest


# These suites exercise Linux-specific CPU affinity, /proc, ELF loader or
# NVIDIA clock/telemetry preparation. The full suite still runs on Linux.
LINUX_DIAGNOSTIC_MODULES = {
    "test_clock_observer_roles",
    "test_clock_role_binding",
    "test_finalize_clock_diagnostic",
    "test_finalize_readonly_trace",
    "test_linux_comparison_plan",
    "test_linux_cpu_telemetry",
    "test_prepare_clock_libraries",
    "test_prepare_clock_preloads",
    "test_prepare_clock_snapshot",
    "test_prepare_clock_upstream",
    "test_prepare_readonly_launch",
    "test_readonly_trace_preflight",
    "test_readonly_trace_runtime",
    "test_readonly_trace_sequence",
    "test_validate_clock_telemetry",
    "test_validate_readonly_telemetry",
}


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def select(discovered):
    return [test for test in discovered if test.id().split(".", 1)[0] not in LINUX_DIAGNOSTIC_MODULES]


def main():
    if sys.platform != "darwin":
        raise SystemExit("macOS report-test selection requires Darwin")
    discovered = list(cases(unittest.defaultTestLoader.discover("tools/bench", "test_*.py")))
    selected = select(discovered)
    excluded = len(discovered) - len(selected)
    print(f"macOS report tests: {len(selected)} selected; {excluded} Linux-diagnostic tests excluded", flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
