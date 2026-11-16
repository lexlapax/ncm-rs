import unittest

from summarize_xsum_gpu_activity import summarize


class DriverMemorySummaryTests(unittest.TestCase):
    def test_attributes_only_owned_processes_in_their_windows(self):
        manifest = {'batches': [1], 'processes_per_runtime_batch': 1}
        events = [
            {'event': 'start', 'name': 'rust-b1-r0', 'pid': 10, 'monotonic': 1},
            {'event': 'end', 'name': 'rust-b1-r0', 'monotonic': 3, 'returncode': 0},
            {'event': 'start', 'name': 'python-b1-r0', 'pid': 20, 'monotonic': 4},
            {'event': 'end', 'name': 'python-b1-r0', 'monotonic': 6, 'returncode': 0},
        ]
        samples = [
            {'monotonic': 2, 'parents': {'11': 10, '99': 98},
             'queries': [{}, {'status': 0, 'stdout': '11, rust, 100\n99, other, 900\n'}]},
            {'monotonic': 5, 'parents': {'21': 20},
             'queries': [{}, {'status': 0, 'stdout': '21, python, 200 MiB\n'}]},
            {'monotonic': 7, 'parents': {}, 'queries': [{}, {'status': 1}]},
        ]
        result = summarize(manifest, events, samples)
        self.assertEqual(result['batches']['1']['rust']['median_sampled_peak_mib'], 100)
        self.assertEqual(result['batches']['1']['python']['median_sampled_peak_mib'], 200)
        self.assertEqual(result['failed_memory_queries'], 1)
        with self.assertRaises(ValueError):
            summarize(manifest, events, samples[:1])


if __name__ == '__main__':
    unittest.main()
