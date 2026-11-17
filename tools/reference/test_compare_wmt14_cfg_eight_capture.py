import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('compare_wmt14_cfg_eight_capture.py')


class MicrobatchComparisonTests(unittest.TestCase):
    def run_comparison(self, internal=8, altered=False):
        row = {'row': 0, 'codes': [0] * 313, 'raw_ids': [1] * 64}
        left = {'batch_size': 4, 'start': 0, 'count': 1, 'cfg_scale': 1.0,
                'cfg_null_eos': 151645, 'outputs': [row]}
        right = json.loads(json.dumps(left))
        right.update(batch_size=16, internal_batch_size=internal)
        if altered:
            right['outputs'][0]['raw_ids'][3] = 2
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('left.json', 'right.json', 'out.json')]
            for path, value in zip(paths, (left, right), strict=False):
                path.write_text(json.dumps(value))
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:2]]
            result = subprocess.run([sys.executable, str(SCRIPT), *map(str, paths),
                                     '--reference-sha256', hashes[0], '--candidate-sha256', hashes[1]],
                                    capture_output=True, text=True, check=False)
            output = json.loads(paths[2].read_text()) if paths[2].exists() else None
            return result.returncode, output

    def test_equal_outputs_pass_and_different_ids_fail(self):
        status, output = self.run_comparison()
        self.assertEqual(status, 0)
        self.assertEqual(output['id_differences'], 0)
        self.assertIn('internal 8', output['scope'])
        status, output = self.run_comparison(altered=True)
        self.assertEqual(status, 1)
        self.assertEqual(output['id_rows'], [0])

    def test_wrong_internal_batch_is_rejected(self):
        status, output = self.run_comparison(internal=4)
        self.assertNotEqual(status, 0)
        self.assertIsNone(output)


if __name__ == '__main__':
    unittest.main()
