import unittest

from phase5_wmt14_cfg1_cli import (DECODE_CAPTURE_SHA, check_output, command_for, expected_batch,
                                     validate_decoded_reference, validate_text_reference)


class WmtCfgCliTests(unittest.TestCase):
    def setUp(self):
        self.prefixes = [[0] * 96 for _ in range(3003)]
        self.reference = {'count': 3003, 'start': 0, 'batch_size': 4, 'cfg_scale': 1.0,
                          'outputs': [{'row': i, 'codes': [0] * 313, 'raw_ids': [1] * 64}
                                      for i in range(3003)]}

    def test_commands_use_explicit_cfg_one_for_each_route(self):
        self.assertEqual(command_for('unified', 'bin', 'art', 'input'),
                         ['bin', 'sample', 'art', 'cuda', 'input', '--cfg', '1'])
        self.assertEqual(command_for('alias', 'bin', 'art', 'input'),
                         ['bin', 'sample-wmt14', 'art', 'cuda', 'input', '1'])
        self.assertEqual(command_for('unified', 'bin', 'art', 'input', text=True)[-3:], ['--cfg', '1', '--text'])
        self.assertEqual(command_for('unified', 'bin', 'art', 'input', text=True, decode=True)[-2:],
                         ['--text', '--decode'])
        for route, text, decode in [('other', False, False), ('alias', True, False), ('alias', False, True)]:
            with self.assertRaises(ValueError):
                command_for(route, 'bin', 'art', 'input', text=text, decode=decode)

    def test_text_reference_requires_exact_prefixes_and_strings(self):
        ref = {'ids': self.prefixes, 'texts': [''] * 3003}
        self.assertEqual(len(validate_text_reference(ref, self.prefixes)), 3003)
        for invalid in ({**ref, 'texts': [''] * 3002}, {**ref, 'ids': []}, {**ref, 'extra': 1},
                        {**ref, 'texts': [True] * 3003}):
            with self.assertRaises(ValueError):
                validate_text_reference(invalid, self.prefixes)

    def test_decoded_reference_requires_expected_capture_and_strings(self):
        ref = {'capture_sha256': DECODE_CAPTURE_SHA, 'start': 0, 'count': 3003, 'hypotheses': ['text'] * 3003}
        self.assertEqual(len(validate_decoded_reference(ref)), 3003)
        for invalid in ({**ref, 'capture_sha256': 'wrong'}, {**ref, 'start': 1},
                        {**ref, 'hypotheses': ['text'] * 3002}, {**ref, 'hypotheses': [True] * 3003}):
            with self.assertRaises(ValueError):
                validate_decoded_reference(invalid)

    def test_full_and_tail_batch(self):
        self.assertEqual(len(expected_batch(self.prefixes, self.reference, 0)[0]), 4)
        self.assertEqual(len(expected_batch(self.prefixes, self.reference, 3000)[0]), 3)
        with self.assertRaises(ValueError):
            expected_batch(self.prefixes, self.reference, 1)

    def test_rejects_changed_reference_and_prefix(self):
        self.reference['outputs'][0]['row'] = 1
        with self.assertRaises(ValueError):
            expected_batch(self.prefixes, self.reference, 0)
        self.reference['outputs'][0]['row'] = 0
        self.prefixes[0][0] = True
        with self.assertRaises(ValueError):
            expected_batch(self.prefixes, self.reference, 0)
        self.prefixes[0][0] = 0
        self.reference['cfg_scale'] = 0
        with self.assertRaises(ValueError):
            expected_batch(self.prefixes, self.reference, 0)

    def test_output_rejects_missing_extra_and_boolean_values(self):
        expected = {'codes': [[0]], 'ids': [[1]]}
        check_output(b'{"codes":[[0]],"ids":[[1]]}', expected)
        check_output(b'{"codes":[[0]],"ids":[[1]],"texts":["x"]}', {**expected, 'texts': ['x']})
        with self.assertRaises(ValueError):
            check_output(b'{"codes":[[0]],"ids":[[1]],"texts":["wrong"]}', {**expected, 'texts': ['x']})
        for raw in (b'{"codes":[[0]]}', b'{"codes":[[0]],"ids":[[1]],"extra":1}',
                    b'{"codes":[[false]],"ids":[[1]]}', b'{"codes":[[1]],"ids":[[1]]}'):
            with self.assertRaises(ValueError):
                check_output(raw, expected)


if __name__ == '__main__':
    unittest.main()
