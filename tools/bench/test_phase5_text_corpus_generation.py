"""Offline controls for corpus text-generation equivalence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_text_corpus_generation as tool


class CorpusGenerationTests(unittest.TestCase):
    def test_batch_plans_cover_every_row_once_and_keep_original_grouping(self):
        for name, expected_batches, size in [('xsum', 356, 32), ('wmt14', 751, 4)]:
            plan = tool.batches(name)
            self.assertEqual(len(plan), expected_batches)
            all_rows = [row for _, _, indices in plan for row in indices]
            self.assertEqual(sorted(all_rows), list(range(tool.PROFILES[name]['rows'])))
            self.assertTrue(all(1 <= len(indices) <= size for _, _, indices in plan))
        self.assertEqual(tool.batches('xsum')[0][2][:4], [0, 2, 4, 6])
        self.assertEqual(tool.batches('xsum')[178][2][:4], [1, 3, 5, 7])
        self.assertEqual(tool.batches('wmt14')[-1][2], [3000, 3001, 3002])

    def test_reference_and_output_validation_fail_closed(self):
        profile = {'rows': 2, 'width': 2, 'code_width': 2, 'id_width': 1}
        ref = {'texts': ['a', 'b'], 'ids': [[1, 2], [3, 4]]}
        tool.validate_reference(ref, profile)
        with self.assertRaises(ValueError):
            tool.validate_reference({'texts': ref['texts'], 'ids': [[True, 2], [3, 4]]}, profile)
        expected = {'codes': [[1, 2], [3, 4]], 'ids': [[5], [6]]}
        tool.validate_output(expected, expected, 2, profile)
        for bad in ({'codes': [[1, 2], [3, 4]], 'ids': [[5], [7]]},
                    {'codes': [[1, 2]], 'ids': [[5]]},
                    {'codes': [[1, 2], [3, 4]], 'ids': [[5], [6]], 'texts': ['a', 'b']}):
            with self.assertRaises(ValueError):
                tool.validate_output(bad, expected, 2, profile)

    def test_existing_output_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'dangling'
            output.symlink_to(Path(directory) / 'missing')
            argv = ['screen', '--profile', 'xsum', '--candidate', 'bin', '--reference', 'ref',
                    '--artifacts', 'artifacts', '--output', str(output)]
            with patch('sys.argv', argv), patch.object(tool, 'read_checked') as checked:
                with self.assertRaisesRegex(ValueError, 'output exists'):
                    tool.main()
                checked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
