"""Run all Linux report tests, overlapping three independently isolated cases.

The three selected tests each perform their work in separate child processes and
use disjoint temporary or read-only inputs. Every other test remains serial.
"""
import concurrent.futures
import io
import sys
import unittest
from collections import Counter

OVERLAPPED = frozenset({
    'test_clock_restoration_capture.RestorationCaptureTests.test_cancelled_supervisor_finishes_twelve_second_cleanup_retry_and_large_output',
    'test_qualify_clock_codec.ClockCodecQualificationTests.test_all_eighteen_archived_outputs_and_rehashed_negatives_in_fresh_child',
    'test_finalize_clock_diagnostic.ClockFinalizationTests.test_failed_gates_preserve_partial_evidence_without_rehash_or_completion',
})


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def partition(discovered):
    tests = list(cases(discovered))
    counts = Counter(test.id() for test in tests)
    if any(counts[name] != 1 for name in OVERLAPPED):
        raise ValueError('each isolated case must occur exactly once')
    serial = [test for test in tests if test.id() not in OVERLAPPED]
    parallel = [test for test in tests if test.id() in OVERLAPPED]
    if len(serial) + len(parallel) != len(tests) or len(parallel) != len(OVERLAPPED):
        raise ValueError('report-test partition is incomplete')
    return serial, parallel


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def startTest(self, test):
        self.executed.append(test.id())
        super().startTest(test)


def run_one(test):
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult).run(unittest.TestSuite([test]))
    return test.id(), result, stream.getvalue()


def main():
    if sys.platform != 'linux':
        raise SystemExit('Linux report-test scheduling requires Linux')
    serial, parallel = partition(unittest.defaultTestLoader.discover('tools/bench', 'test_*.py'))
    expected_ids = Counter(test.id() for test in serial + parallel)
    serial_result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(unittest.TestSuite(serial))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(OVERLAPPED)) as pool:
        results = list(pool.map(run_one, parallel))
    for _, _, output in results:
        sys.stderr.write(output)
    count = serial_result.testsRun + sum(result.testsRun for _, result, _ in results)
    expected = len(serial) + len(parallel)
    print(f'Linux report tests: {count}/{expected} executed; three isolated cases overlapped', flush=True)
    actual_ids = Counter(serial_result.executed)
    for _, result, _ in results:
        actual_ids.update(result.executed)
    if (count != expected or actual_ids != expected_ids or not serial_result.wasSuccessful()
            or any(not result.wasSuccessful() for _, result, _ in results)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
