"""Offline controls for the sanitized text cold-screen exporter."""
import unittest

import export_phase5_text_cold as tool


def records():
    rows = []
    for group in range(1, 4):
        for ordinal in range(35):
            for condition in tool.screen.order(group, ordinal):
                rows.append({'group': group, 'ordinal': ordinal, 'condition': condition,
                             'warmup': ordinal < 5, 'elapsed_seconds': float(ordinal + 1) + tool.screen.CONDITIONS.index(condition),
                             'max_rss_kib': 100 + ordinal})
    return rows


class TextColdExportTests(unittest.TestCase):
    def test_paired_statistics(self):
        rows = records()
        self.assertEqual(tool.screen.summarize(rows)['old_id']['samples'], 90)
        pairs = tool.paired(rows)
        self.assertEqual(pairs['new_text_minus_new_id']['median_seconds'], 1.)
        self.assertEqual(pairs['new_text_minus_new_id']['bootstrap_95_seconds'], [1., 1.])

    def test_missing_or_duplicate_pairs_rejected(self):
        rows = records()
        rows.pop()
        with self.assertRaises(ValueError):
            tool.screen.summarize(rows)
        rows = records()
        rows[-1]['condition'] = rows[-2]['condition']
        with self.assertRaises(ValueError):
            tool.screen.summarize(rows)
        with self.assertRaises(ValueError):
            tool.paired(rows)


if __name__ == '__main__':
    unittest.main()
