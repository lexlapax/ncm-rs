import unittest

from phase5_wmt14_cfg1_cli import check_output, command_for, expected_batch


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
        with self.assertRaises(ValueError):
            command_for('other', 'bin', 'art', 'input')

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
        for raw in (b'{"codes":[[0]]}', b'{"codes":[[0]],"ids":[[1]],"extra":1}',
                    b'{"codes":[[false]],"ids":[[1]]}', b'{"codes":[[1]],"ids":[[1]]}'):
            with self.assertRaises(ValueError):
                check_output(raw, expected)


if __name__ == '__main__':
    unittest.main()
