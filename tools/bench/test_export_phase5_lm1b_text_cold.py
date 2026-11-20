"""Offline auditing controls for LM1B cold-text exports."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import export_phase5_lm1b_text_cold as tool
from test_export_phase5_text_cold import records


class Lm1bTextExportTests(unittest.TestCase):
    def test_full_audit_rejects_tampered_stdout_and_rss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = records()
            hashes = (hashlib.sha256(b'old').hexdigest(), hashlib.sha256(b'decoded').hexdigest())
            for row in rows:
                stem = f"{row['group']}-{row['ordinal']:02}-{row['condition']}"
                raw = b'decoded' if row['condition'] == 'new_text_decode' else b'old'
                row['output_sha256'] = hashlib.sha256(raw).hexdigest()
                (root / f'{stem}.stdout').write_bytes(raw)
                (root / f'{stem}.rss').write_text(str(row['max_rss_kib']))
            gpu = b'gpu'; (root / 'gpu.csv').write_bytes(gpu)
            data = {'size': 'small', 'gpu_csv_sha256': tool.digest(gpu), 'records': rows,
                    'summary': tool.lm.cold.summarize(rows)}
            raw = json.dumps(data).encode(); (root / 'result.json').write_bytes(raw)
            with patch.dict(tool.PINS, {'small': (tool.digest(raw), tool.digest(gpu))}), \
                 patch.dict(tool.lm.HASHES, {'small': hashes}):
                result = tool.audit_run('small', root)
                self.assertEqual(len(result['groups']), 3)
                (root / '1-00-old_id.stdout').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'stdout differs'):
                    tool.audit_run('small', root)
                (root / '1-00-old_id.stdout').write_bytes(b'old')
                (root / '1-00-old_id.rss').write_text('1')
                with self.assertRaisesRegex(ValueError, 'RSS differs'):
                    tool.audit_run('small', root)


if __name__ == '__main__':
    unittest.main()
