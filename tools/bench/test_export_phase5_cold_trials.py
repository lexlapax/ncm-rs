"""Controls for sanitized cold-request trial exports."""
import copy
import json
import unittest

import export_phase5_cold_trials as tool


def records():
    return [{'group': group, 'ordinal': ordinal, 'condition': condition, 'warmup': ordinal < 5,
             'elapsed_seconds': float(ordinal + 1), 'max_rss_kib': ordinal + 100,
             'output': 'private-evaluation-content', 'path': '/secret/path'}
            for group in range(1, 4) for ordinal in range(35) for condition in ('baseline', 'candidate')]


class ColdTrialExportTests(unittest.TestCase):
    def test_measured_pairs_and_privacy(self):
        groups = tool.extract(records())
        self.assertEqual(groups[0]['baseline']['elapsed_seconds'][0], 6.)
        self.assertEqual(groups[-1]['candidate']['max_rss_kib'][-1], 134)
        values = tool.summarize(groups)
        self.assertEqual(values['baseline']['median_seconds'], 20.5)
        self.assertEqual(values['candidate']['p95_seconds_nearest_rank'], 34.)
        encoded = json.dumps(groups)
        self.assertNotIn('private-evaluation-content', encoded)
        self.assertNotIn('/secret/path', encoded)

    def test_verification_rejects_changed_trials_and_extra_fields(self):
        groups = tool.extract(records())
        values = tool.summarize(groups)
        aggregate = {'profiles': {name: {'raw_result_sha256': 'a', 'gpu_csv_sha256': 'b', **values}
                                  for name in tool.audit.PROFILES}}
        exported = {'schema_version': 1, 'scope': 'scope', 'aggregate_sha256': tool.AGGREGATE_SHA,
                    'warmup_trials_omitted': True, 'pair_order': 'order',
                    'profiles': {name: {'raw_result_sha256': 'a', 'gpu_csv_sha256': 'b', 'groups': copy.deepcopy(groups)}
                                 for name in tool.audit.PROFILES}}
        tool.verify_export(exported, aggregate)
        bad = copy.deepcopy(exported)
        bad['profiles']['xsum']['groups'][0]['baseline']['elapsed_seconds'][0] = 9999.
        with self.assertRaises(ValueError):
            tool.verify_export(bad, aggregate)
        bad = copy.deepcopy(exported)
        bad['profiles']['xsum']['raw_output'] = 'private'
        with self.assertRaises(ValueError):
            tool.verify_export(bad, aggregate)

    def test_rejects_missing_duplicates_or_invalid_measurements(self):
        for mutate in (lambda rows: rows.pop(),
                       lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                       lambda rows: rows[10].update(warmup=True),
                       lambda rows: rows[10].update(elapsed_seconds=float('nan')),
                       lambda rows: rows[10].update(max_rss_kib=True)):
            rows = records()
            mutate(rows)
            with self.assertRaises(ValueError):
                tool.extract(rows)
        groups = tool.extract(records())
        groups[0]['baseline']['elapsed_seconds'].pop()
        with self.assertRaises(ValueError):
            tool.summarize(groups)


if __name__ == '__main__':
    unittest.main()
