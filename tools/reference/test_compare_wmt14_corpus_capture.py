import unittest

from compare_wmt14_corpus_capture import ROWS, compare_batches


class Wmt14CorpusCaptureTests(unittest.TestCase):
    def setUp(self):
        self.reference = [
            {'row': offset, 'codes': [0] * 313, 'raw_ids': [0] * 64} for offset in range(ROWS)
        ]

    def batch(self, start, count):
        return {'start': start, 'count': count, 'codes': [[0] * 313 for _ in range(count)],
                'raw_ids': [[0] * 64 for _ in range(count)]}

    def test_retains_rejection_retry_and_token_differences(self):
        rejected = {'start': 2996, 'count': 4, 'error': 'head rejected'}
        recovered = self.batch(3000, 3)
        recovered['internal_retry_batch'] = 8
        recovered['codes'][2][0] = 7
        recovered['raw_ids'][1][0] = 9
        result = compare_batches(self.reference, [rejected, recovered], 2996, 7)
        self.assertEqual(result['rejected_batches'], [rejected])
        self.assertEqual(result['retried_batches'], [3000])
        self.assertEqual(result['rows_compared'], 3)
        self.assertEqual((result['code_differences'], result['code_rows']), (1, [3002]))
        self.assertEqual((result['id_differences'], result['id_rows']), (1, [3001]))

    def test_rejects_missing_reordered_and_invalid_records(self):
        first, last = self.batch(2996, 4), self.batch(3000, 3)
        for records in ([first], [last, first], [first, last, last], [first, dict(last, count=True)],
                        [first, dict(last, internal_retry_batch=8.0)]):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                compare_batches(self.reference, records, 2996, 7)
        last['codes'][0][1] = 4096
        with self.assertRaises(ValueError):
            compare_batches(self.reference, [first, last], 2996, 7)


if __name__ == '__main__':
    unittest.main()
