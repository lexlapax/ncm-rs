"""Offline controls for bounded text/ID generation comparisons."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_text_generation_screen as tool


class TextGenerationScreenTests(unittest.TestCase):
    def test_reference_shape_and_fixed_selection(self):
        profile = {'rows': 1300, 'width': 2}
        reference = {'texts': ['x'] * 1300, 'ids': [[1, 2]] * 1300}
        self.assertEqual([len(x) for x in tool.select(reference, profile)], [1, 4, 32])
        self.assertEqual(tool.select(reference, profile)[-1][-1], 1299)
        reference['ids'][-1] = [True, 2]
        with self.assertRaisesRegex(ValueError, 'invalid IDs'):
            tool.select(reference, profile)

    def test_output_comparison_rejects_differences_and_bad_shapes(self):
        profile = {'codes': 2, 'ids': 1}
        raw = json.dumps({'codes': [[1, 2]], 'ids': [[3]]}).encode()
        decoded = json.dumps({'codes': [[1, 2]], 'ids': [[3]], 'texts': ['x']}).encode()
        tool.validate_pair(raw, decoded, decoded, 1, profile)
        for old, a, b in [(raw, decoded, decoded + b' '),
                          (raw, decoded, b'{"codes":[[1,2]],"ids":[[4]],"texts":["x"]}'),
                          (raw, decoded, decoded)]:
            with self.assertRaises(ValueError):
                tool.validate_pair(old, a, b, 2 if a == b else 1, profile)

    def test_existing_output_and_symlink_rejected_before_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing'
            existing.mkdir()
            dangling = root / 'dangling'
            dangling.symlink_to(root / 'missing')
            for output in (existing, dangling):
                argv = ['screen', '--baseline', 'old', '--candidate', 'new', '--references', 'ref',
                        '--artifacts', 'art', '--output', str(output)]
                with patch('sys.argv', argv), patch.object(tool, 'checked') as checked:
                    with self.assertRaisesRegex(ValueError, 'output exists'):
                        tool.main()
                    checked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
