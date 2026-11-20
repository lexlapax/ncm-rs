import json
import unittest

import export_phase5_tinystories_trials as tool


class TinyTrialExportTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'batches': list(tool.BATCHES), 'processes_per_runtime_batch': 3}
        self.summary = {'batches': {str(batch): {runtime: {'process_medians_ms': [100.0] * 3}
                                                   for runtime in ('python', 'rust')}
                                    for batch in tool.BATCHES}}
        self.records = {(runtime, batch, rep): json.dumps({'batch': batch, 'warmups': 5,
                                                            'trials_seconds': [.1] * 30,
                                                            'outputs': ['private-input-must-not-export']}).encode()
                        for batch in tool.BATCHES for runtime in ('python', 'rust') for rep in range(3)}

    def extract(self):
        return tool.extract(self.manifest, self.summary, lambda *key: self.records[key])

    def test_extracts_only_sanitized_timing(self):
        out = self.extract()
        self.assertNotIn('private-input-must-not-export', json.dumps(out))
        self.assertEqual(len(out['batches']['128']['rust'][2]['trials_seconds']), 30)
        self.assertEqual(set(out['batches']['1']['python'][0]), {'trials_seconds', 'source_sha256'})

    def test_rejects_missing_changed_nonfinite_and_boolean_values(self):
        key = ('rust', 1, 0)
        for values in ([.1] * 29, [.1] * 29 + [float('nan')], [.1] * 29 + [True], [.2] * 30):
            self.records[key] = json.dumps({'batch': 1, 'warmups': 5, 'trials_seconds': values}).encode()
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.extract()
        del self.records[key]
        with self.assertRaises(KeyError):
            self.extract()


if __name__ == '__main__':
    unittest.main()
