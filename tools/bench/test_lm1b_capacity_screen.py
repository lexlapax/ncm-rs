"""Regression tests for capacity evidence aggregation."""
import unittest
from run_lm1b_small_capacity_screen import summarize


def rows():
    return [{'runtime': runtime, 'batch': batch, 'rep': rep,
             'signatures': ['a' * 64] * 8, 'seconds': [float(batch)] * 8}
            for batch in (1, 4, 16) for runtime in ('python', 'rust') for rep in range(3)]


class CapacitySummaryTests(unittest.TestCase):
    def test_complete_evidence_and_throughput(self):
        result = summarize(rows())
        self.assertEqual(result['4']['rust']['median_seconds'], 4.0)
        self.assertEqual(result['4']['rust']['samples_per_second_at_median'], 1.0)
        self.assertEqual(result['16']['python']['samples'], 24)

    def test_repeated_evidence_requires_all_trials_per_process(self):
        repeated = [dict(row, seconds=[float(row['batch'])] * 30) for row in rows()]
        self.assertEqual(summarize(repeated, 30)['16']['rust']['samples'], 90)
        repeated[0]['seconds'].pop()
        repeated[1]['seconds'].append(1.0)
        with self.assertRaisesRegex(ValueError, 'invalid timings'):
            summarize(repeated, 30)

    def test_missing_duplicate_changed_and_nonfinite_fail(self):
        data = rows()
        for invalid in (data[:-1], data + [data[0]], [dict(x, rep=0) if i == 1 else x for i, x in enumerate(data)],
                        [dict(x, signatures=['b' * 64] * 8) if i == 1 else x for i, x in enumerate(data)],
                        [dict(x, seconds=[float('nan')] * 8) if i == 1 else x for i, x in enumerate(data)]):
            with self.subTest(size=len(invalid)):
                with self.assertRaises(ValueError):
                    summarize(invalid)


if __name__ == '__main__':
    unittest.main()
