"""The CUDA driver may report fractional GPU temperatures."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SummaryTest(unittest.TestCase):
    def test_fractional_temperature_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = []
            for runtime in ('rust', 'python'):
                for rep in range(3):
                    result = {'trials_seconds': [1.0] * 30}
                    if runtime == 'rust':
                        result['cuda_default_pool'] = {}
                    else:
                        result.update(peak_allocated_bytes=1, peak_reserved_bytes=2,
                                      cleanup_seconds=[0.0] * 30)
                    (root / f'{runtime}-b1-r{rep}.json').write_text(json.dumps(result))
                    (root / f'{runtime}-b1-r{rep}.time').write_text('Maximum resident set size (kbytes): 1024\n')
                    events.extend([{'event': 'start', 'pid': len(events) + 1},
                                   {'event': 'end', 'returncode': 0}])
            (root / 'process-events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
            sample = {'queries': [{'status': 0, 'stdout': 'now, 10, 20, 71.97, 200, 1500, 6800'},
                                  {'status': 0, 'stdout': ''}], 'parents': {}}
            (root / 'activity.jsonl').write_text(json.dumps(sample) + '\n')
            script = Path(__file__).with_name('summarize_phase4_xsum_b32.py')
            result = subprocess.run([sys.executable, str(script), '--directory', str(root), '--batches', '1'],
                                    capture_output=True, text=True, check=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary['activity']['gpu_temperature_c_range'], [71.97, 71.97])


if __name__ == '__main__':
    unittest.main()
