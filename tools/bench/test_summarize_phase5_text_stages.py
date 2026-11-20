"""Controls for tokenizer stage diagnostic aggregation."""
import json
import unittest

import summarize_phase5_text_stages as tool


class TextStageSummaryTests(unittest.TestCase):
    def fixture(self):
        record = {'profile': 'xsum', 'warmups': 5, 'trials': 30,
                  **{name: [0.1] * 30 for name in tool.STAGES}}
        log = ('TEXT_STAGE_TIMINGS ' + json.dumps(record) + '\ntest result: ok. 1 passed;\n').encode()
        return record, log, b'Maximum resident set size (kbytes): 1234\n'

    def test_valid(self):
        record, log, time_file = self.fixture()
        actual, rss = tool.parse('xsum', log, time_file)
        self.assertEqual(actual, record)
        self.assertEqual(rss, 1234)

    def test_rejects_wrong_profile_missing_success_and_invalid_samples(self):
        record, log, time_file = self.fixture()
        with self.assertRaises(ValueError):
            tool.parse('wmt14', log, time_file)
        with self.assertRaises(ValueError):
            tool.parse('xsum', log.replace(b'test result: ok.', b'test result: FAILED.'), time_file)
        with self.assertRaises(ValueError):
            tool.parse('xsum', log, b'no rss')
        record['construct_seconds'][0] = float('nan')
        with self.assertRaises(ValueError):
            tool.parse('xsum', ('TEXT_STAGE_TIMINGS ' + json.dumps(record) + '\ntest result: ok. 1 passed;').encode(), time_file)


if __name__ == '__main__':
    unittest.main()
