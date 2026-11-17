import unittest

from lm1b_small_teacher_trace import fixed_codes


class Lm1bSmallTeacherTraceTests(unittest.TestCase):
    def test_fixed_codes_validate_metadata_and_all_rows(self):
        data = {'size': 'small', 'seed': 11, 'count': 32, 'batch_size': 8,
                'codes': [[0] * 641 for _ in range(32)]}
        self.assertEqual(fixed_codes(data), data['codes'][:2])
        data['codes'][10][0] = 8192
        with self.assertRaises(ValueError):
            fixed_codes(data)
        data['codes'][10][0] = 0
        data['seed'] = 42
        with self.assertRaises(ValueError):
            fixed_codes(data)


if __name__ == '__main__':
    unittest.main()
