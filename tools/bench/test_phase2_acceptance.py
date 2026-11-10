import copy
import json
from pathlib import Path
import unittest
import tarfile
import tempfile
from unittest.mock import patch
from summarize_codec_load_control import summarize as codec_load_control
from audit_phase2_performance import quality
from run_codec_lifecycle import validate

ROOT=Path(__file__).resolve().parents[2]


class AcceptanceTests(unittest.TestCase):
    def test_quality_is_not_an_empty_or_relaxed_gate(self):
        raw=json.loads((ROOT/'docs/results/2026-11-10-phase2-cuda-remediation/raw/qualification.json').read_bytes())
        quality(raw)
        for mutate in [lambda q:q.update(traces=[]),lambda q:q.update(all_passed=False),
                       lambda q:q.update(left_trace_sha256='wrong'),lambda q:q['policy']['primitive'].update(atol=1),
                       lambda q:q['traces'][0].update(mismatches=1),lambda q:q.update(selections={})]:
            q=copy.deepcopy(raw);mutate(q)
            with self.assertRaises(ValueError):quality(q)

    def test_codec_load_control_validates_full_reports(self):
        archive=ROOT/'docs/results/2026-11-10-phase2-shared-loading/raw/linux-token-buffer-v1.tar.gz'
        with tempfile.TemporaryDirectory() as tmp, patch('summarize_codec_load_control.interval', return_value={}):
            root=Path(tmp)
            with tarfile.open(archive) as bundle: bundle.extractall(root, filter='data')
            result=codec_load_control(root,'cuda')
            self.assertEqual(result['measurements'][0]['scope'],'checked_load')
            path=root/'candidate/codec-rust-1.json';original=path.read_bytes()
            for mutate in [lambda r:r.update(load_trials_ms=[1.]), lambda r:r.update(runtime='wrong'),
                           lambda r:r.update(diagnostic_profiling=True), lambda r:r['measurements'][1]['first_logits'].__setitem__(0,1e10)]:
                raw=json.loads(original);mutate(raw);path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):codec_load_control(root,'cuda')
                path.write_bytes(original)
            path.unlink()
            with self.assertRaises(ValueError):codec_load_control(root,'cuda')

    def test_lifecycle_requires_every_phase_and_identity(self):
        rows=[{'schema_version':1,'phase':p,'internal_ms':1.,'output_ids_sha256':'a'*64 if p=='execute' else None}
              for p in ('context','load','execute','drop')]
        self.assertEqual(validate(rows),'a'*64)
        with self.assertRaises(ValueError):validate(rows[:-1])
        with self.assertRaises(ValueError):validate(rows[::-1])
        bad=copy.deepcopy(rows);bad[2]['output_ids_sha256']=None
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(rows);bad[0]['internal_ms']=float('nan')
        with self.assertRaises(ValueError):validate(bad)


if __name__=='__main__':unittest.main()
