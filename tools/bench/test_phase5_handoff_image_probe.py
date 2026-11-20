import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import observe_readonly_process as observer

import phase5_handoff_image_probe as probe


class HandoffImageProbeTests(unittest.TestCase):
    def test_categories_serialize_only_fixed_booleans(self):
        result = probe.categories([(12, 34), (56, 78)], (12, 34), (56, 78), (90, 11))
        self.assertEqual(result['samples'][0], {'collector': True, 'workload': False, 'launcher': False})
        self.assertEqual(result['samples'][1], {'collector': False, 'workload': True, 'launcher': False})
        self.assertNotIn('34', json.dumps(result))
        self.assertNotIn('78', json.dumps(result))

    def test_instrumentation_preserves_mismatch_and_restores_stat(self):
        def failing(_self, pid):
            os.stat(f'/proc/{pid}/exe')
            os.stat(f'/proc/{pid}/exe')
            raise observer.ObservationMismatch('unexpected live executable identity')
        image = os.stat('/proc/self/exe')
        fake = SimpleNamespace(workload={'device': image.st_dev, 'inode': image.st_ino}, launcher=None)
        original_stat = os.stat
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'category.jsonl'
            with patch.object(observer.ProcessObservation, 'observe', failing):
                probe.install(target)
                with self.assertRaisesRegex(observer.ObservationMismatch, 'unexpected live executable identity'):
                    observer.ProcessObservation.observe(fake, os.getpid())
                self.assertIs(os.stat, original_stat)
            sample = json.loads(target.read_text())['samples']
            self.assertEqual(len(sample), 2)
            self.assertTrue(all(row['collector'] and row['workload'] for row in sample))

    def test_symlink_is_rejected_before_running_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'link'
            output.symlink_to(Path(directory) / 'missing')
            with patch('sys.argv', ['probe', '--output', str(output)]), patch.object(probe.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'new output'):
                    probe.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
