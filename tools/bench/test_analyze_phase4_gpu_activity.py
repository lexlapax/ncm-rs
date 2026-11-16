import unittest

from analyze_phase4_gpu_activity import analyze


class ActivityAlignmentTests(unittest.TestCase):
    def setUp(self):
        base = 1_700_000_000_000_000_000
        self.report = {'warmups': 5, 'trials_seconds': [1.0] * 30,
                       'request_wall_ns': [[base + i * 2_000_000_000,
                                            base + i * 2_000_000_000 + 900_000_000,
                                            base + i * 2_000_000_000 + 1_000_000_000]
                                           for i in range(35)]}
        observed = self.report['request_wall_ns'][5][0] + 500_000_000
        self.samples = [{'begin_unix_ns': observed - 1000, 'end_unix_ns': observed,
                         'statistics': {'Device Utilization %': 91}, 'last_submission_pid': 7}]

    def test_aligns_sample_only_to_its_request(self):
        result = analyze(self.report, self.samples)
        self.assertEqual(result['requests'][0]['reported_device_utilization'], [91])
        self.assertEqual(result['requests_without_gpu_snapshot'], 29)
        self.assertEqual(result['requests_over_4_seconds'], 0)

    def test_rejects_missing_intervals_and_clock_disagreement(self):
        self.report['request_wall_ns'].pop()
        with self.assertRaises(ValueError):
            analyze(self.report, self.samples)
        self.setUp()
        self.report['trials_seconds'][0] = 2.0
        with self.assertRaises(ValueError):
            analyze(self.report, self.samples)


if __name__ == '__main__':
    unittest.main()
