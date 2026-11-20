"""Offline negative controls for LM1B performance aggregates."""
import copy
import unittest

import summarize_phase5_lm1b as tool


def fixture():
    manifest = {'size': 'small', 'head': 'head', 'batches': [1], 'warmups': 5, 'trials': 30,
                'repetitions': 3, 'rust_convolution_math': 'cuDNN graph and efficient attention'}
    rows, activity = [], []
    for runtime, base in [('python', 10), ('rust', 20)]:
        for rep in range(3):
            pid = base + rep
            native = ({'used_peak': 1, 'reserved_peak': 2} if runtime == 'rust' else
                      {'torch_peak_allocated_bytes': 1, 'torch_peak_reserved_bytes': 2})
            rows.append({'runtime': runtime, 'batch': 1, 'rep': rep, 'pid': pid,
                         'start_monotonic': pid, 'end_monotonic': pid + .9,
                         'load_seconds': 1., 'peak_rss_kib': 1024, 'native_allocator_bytes': native,
                         'seconds': [.1] * 30, 'signatures': ['x'] * 30})
            activity.append({'monotonic': pid + .5, 'queries': [{'status': 0},
                {'status': 0, 'stdout': f'{pid}, ncm, 50 MiB\n9, desktop, 10 MiB'}]})
    result = {'rows': rows, 'summary': tool.summarize(rows, trials=30, batches=(1,))}
    return manifest, result, activity


class Lm1bAggregateTests(unittest.TestCase):
    def test_aggregate_scopes_and_process_attribution(self):
        result = tool.aggregate_run(*fixture())
        self.assertEqual(result['other_process_count_range'], [1, 1])
        self.assertEqual(result['batches']['1']['rust']['sampled_gpu_peak_mib'], [50, 50, 50])
        self.assertEqual(result['batches']['1']['rust']['peak_rss_mib'], [1, 1, 1])

    def test_missing_or_invalid_data_fail(self):
        for mutate in (lambda m, r, a: m.update(warmups=2),
                       lambda m, r, a: r['rows'].pop(),
                       lambda m, r, a: r['summary']['1']['rust'].update(samples=1),
                       lambda m, r, a: r['rows'][0]['native_allocator_bytes'].clear(),
                       lambda m, r, a: a[0]['queries'][1].update(status=1)):
            data = copy.deepcopy(fixture())
            mutate(*data)
            with self.assertRaises(ValueError):
                tool.aggregate_run(*data)


if __name__ == '__main__':
    unittest.main()
