import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import compare_wmt14_corpus_capture as corpus
import replay_wmt14_corpus_quality as replay


def fixture(rejected=False):
    ref = {'outputs': [{'row': i, 'codes': [0] * 313, 'raw_ids': [0] * 64} for i in range(7)]}
    first = {'start': 0, 'count': 4, 'codes': [[0] * 313 for _ in range(4)], 'raw_ids': [[0] * 64] * 4}
    first['codes'][0][-1] = 1
    last = {'start': 4, 'count': 3, 'internal_retry_batch': 8, 'codes': [[0] * 313] * 3,
            'raw_ids': [[0] * 64] * 3}
    if rejected:
        last = {'start': 4, 'count': 3, 'error': 'head rejected'}
    capture = ''.join(json.dumps(record) + '\n' for record in (first, last)).encode()
    return capture, ref


class QualityReplayTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for module in (corpus, replay):
            mocked = patch.object(module, 'ROWS', 7)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_quality_receipts_keep_retries_and_code_differences(self):
        capture, ref = fixture()
        digest = replay.write_receipts(capture, ref, 1, 'metal', self.root)
        batch = json.loads((self.root / 'batch-comparison.json').read_bytes())
        comparison = json.loads((self.root / 'comparison.json').read_bytes())
        normalized = (self.root / 'capture.json').read_bytes()
        self.assertEqual(batch['retried_batches'], [4])
        self.assertEqual(batch['code_rows'], [0])
        self.assertEqual(comparison['code_rows'], [0])
        self.assertEqual(comparison['id_rows'], [])
        self.assertEqual(comparison['candidate_sha256'], replay.digest(normalized))
        self.assertEqual(digest, replay.digest(normalized))
        self.assertEqual(json.loads(normalized)['cfg_scale'], 1.0)

    def test_rejected_batch_is_reported_before_decoding(self):
        capture, ref = fixture(rejected=True)
        self.assertIsNone(replay.write_receipts(capture, ref, 0, 'metal', self.root))
        receipt = json.loads((self.root / 'batch-comparison.json').read_bytes())
        self.assertEqual(receipt['rejected_batches'], [{'start': 4, 'count': 3, 'error': 'head rejected'}])
        self.assertFalse((self.root / 'capture.json').exists())
        self.assertFalse((self.root / 'comparison.json').exists())

    def test_replay_requires_matching_independent_scorer_receipts(self):
        capture, ref = fixture()
        raw, reference, text, out = [self.root / name for name in ('raw', 'reference', 'text', 'out')]
        raw.write_bytes(capture)
        reference.write_text(json.dumps(ref))
        text.write_text('frozen text fixture')

        def run(command, **_):
            if command[1].name == 'wmt14_decode_capture.py':
                command[3].write_text('{}')
            else:
                command[5].write_text(json.dumps({'full': {'all_passed': True}, 'heldout': {'all_passed': True},
                                                  'variant': command[0].parts[-3]}))
            return subprocess.CompletedProcess(command, 0)

        with (patch.dict(replay.REFERENCE_SHA256, {0: replay.digest(reference.read_bytes())}),
              patch.dict(replay.REFERENCE_TEXT_SHA256, {0: replay.digest(text.read_bytes())}),
              patch.object(replay.subprocess, 'run', side_effect=run) as runner,
              self.assertRaisesRegex(ValueError, 'independent WMT14 scorer receipts differ')):
            replay.replay(raw, reference, text, 0, 'cuda', out)
        self.assertEqual(runner.call_count, 3)
        self.assertTrue((out / 'batch-comparison.json').is_file())


if __name__ == '__main__':
    unittest.main()
