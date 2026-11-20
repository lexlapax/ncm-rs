"""Offline integrity checks for the WMT14 timing aggregate."""
import copy
import unittest

import summarize_wmt14_phase5 as tool


def fixture():
    manifest = {'batches': list(tool.BATCHES), 'modes': [0], 'repetitions': 3, 'warmups': 5, 'trials': 30}
    measurements, events, activity = {}, [], []
    pid = 100
    for batch in tool.BATCHES:
        for runtime in ('rust', 'python'):
            records = []
            for rep in range(3):
                name = f'{runtime}-cfg0-b{batch}-r{rep}'
                events.extend([{'event': 'start', 'name': name, 'pid': pid, 'monotonic': pid},
                               {'event': 'end', 'name': name, 'returncode': 0, 'monotonic': pid + .9}])
                activity.append({'monotonic': pid + .5, 'parents': {str(pid + 1000): pid},
                                 'queries': [{'status': 0}, {'status': 0, 'stdout': f'{pid + 1000}, ncm, 200 MiB\n9, desktop, 10 MiB'}]})
                records.append({'trials_seconds': [.1] * 30, 'device_event_seconds': [.09] * 30,
                                'peak_rss_kib': 2048, 'peak_allocated_bytes': None, 'peak_reserved_bytes': None})
                pid += 1
            measurements[f'cfg0-b{batch}-{runtime}'] = records
    return manifest, measurements, events, activity


class AggregateTests(unittest.TestCase):
    def test_complete_and_child_process_attribution(self):
        result = tool.summarize(*fixture())
        self.assertEqual(result['activity_samples'], 24)
        self.assertEqual(result['other_process_count_range'], [1, 1])
        self.assertEqual(result['batches']['1']['rust']['sampled_gpu_peak_mib'], [200, 200, 200])
        self.assertEqual(result['batches']['32']['python']['pooled_p95_ms'], 100)

    def test_missing_and_failed_evidence_rejected(self):
        for mutate in (
            lambda m, v, e, a: m.update(modes=[1]),
            lambda m, v, e, a: e.pop(),
            lambda m, v, e, a: e[1].update(returncode=1),
            lambda m, v, e, a: a[0]['queries'][1].update(status=1),
            lambda m, v, e, a: a.pop(),
            lambda m, v, e, a: v['cfg0-b1-rust'][0]['trials_seconds'].pop(),
            lambda m, v, e, a: v['cfg0-b1-rust'][0]['trials_seconds'].__setitem__(0, float('nan')),
        ):
            data = copy.deepcopy(fixture())
            mutate(*data)
            with self.assertRaises(ValueError):
                tool.summarize(*data)


if __name__ == '__main__':
    unittest.main()
