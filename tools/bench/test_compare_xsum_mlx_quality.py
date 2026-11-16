"""Check receipt validation and separate code/ID disagreement reporting."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compare_xsum_mlx_quality import canonical, compare, envelope


class MetalQualityComparisonTests(unittest.TestCase):
    def test_receipt_rejects_mutation(self):
        record = {'start': 0, 'count': 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'batch.json'
            path.write_text(json.dumps({'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}))
            self.assertEqual(envelope(path), record)
            path.write_text(json.dumps({'record': {'start': 1, 'count': 1},
                                        'sha256': hashlib.sha256(canonical(record)).hexdigest()}))
            with self.assertRaises(ValueError):
                envelope(path)

    def test_distinguishes_code_and_output_disagreements(self):
        baseline = [{'codes': [1, 2], 'ids': [3, 4]}, {'codes': [1, 2], 'ids': [3, 4]}]
        metal = [{'codes': [1, 5], 'ids': [3, 4]}, {'codes': [1, 2], 'ids': [3, 5]}]
        result = compare(metal, baseline)
        self.assertEqual((result['code_differences'], result['id_differences']), (1, 1))
        self.assertEqual((result['code_rows'], result['id_rows']), ([0], [1]))
        with self.assertRaises(ValueError):
            compare(metal, baseline[:1])


if __name__ == '__main__':
    unittest.main()
