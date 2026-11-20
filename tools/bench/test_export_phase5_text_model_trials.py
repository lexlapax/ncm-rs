import json
import unittest

import export_phase5_text_model_trials as tool


class TextModelTrialsTests(unittest.TestCase):
    def fixture(self, name):
        batches = tool.PROFILES[name]['batches']
        manifest = {'batches': list(batches)}
        if name == 'xsum':
            manifest.update(processes_per_runtime_batch=3, warmup_per_process=5, trials_per_process=30)
        else:
            manifest.update(repetitions=3, warmups=5, trials=30, modes=[0])
        summary = {'batches': {str(b): {r: {'process_medians_ms': [100.0] * 3} for r in ('rust', 'python')}
                               for b in batches}}
        records = {(r, b, n): json.dumps({'batch': b, 'warmups': 5, 'cfg': 0, 'cfg_scale': 0.0,
                                          'trials_seconds': [.1] * 30, 'outputs': ['private']}).encode()
                   for b in batches for r in ('rust', 'python') for n in range(3)}
        return manifest, summary, records

    def test_both_profiles_strip_outputs(self):
        for name in tool.PROFILES:
            manifest, summary, records = self.fixture(name)
            result = tool.extract(name, manifest, summary, lambda *k: records[k])
            self.assertNotIn('private', json.dumps(result))
            self.assertEqual(len(result['batches']['1']['rust'][0]['trials_seconds']), 30)

    def test_wrong_cfg_protocol_or_median_fails(self):
        for name in tool.PROFILES:
            manifest, summary, records = self.fixture(name)
            key = ('rust', 1, 0)
            doc = json.loads(records[key])
            for invalid in (1, False):
                doc['cfg' if name == 'wmt14' else 'cfg_scale'] = invalid
                records[key] = json.dumps(doc).encode()
                with self.subTest(profile=name, invalid=invalid), self.assertRaises(ValueError):
                    tool.extract(name, manifest, summary, lambda *k: records[k])
            manifest, summary, records = self.fixture(name)
            summary['batches']['1']['rust']['process_medians_ms'][0] = 101
            with self.assertRaises(ValueError):
                tool.extract(name, manifest, summary, lambda *k: records[k])
            manifest['batches'] = [1]
            with self.assertRaises(ValueError):
                tool.extract(name, manifest, summary, lambda *k: records[k])


if __name__ == '__main__':
    unittest.main()
