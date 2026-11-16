"""Focused integrity checks for stochastic quality analysis."""
import json
import tempfile
import unittest
from pathlib import Path

from xsum_stochastic_analyze import freeze, load, metrics
from xsum_stochastic_capture import DATA_SHA, ROOT, validate_protocol, valid_rows


class StochasticAnalysisTest(unittest.TestCase):
    def test_bounds_and_boolean_are_rejected(self):
        self.assertTrue(valid_rows([[0, 3]], 1, 2, 4))
        self.assertFalse(valid_rows([[0, 4]], 1, 2, 4))
        self.assertFalse(valid_rows([[True, 1]], 1, 2, 4))

    def test_frozen_protocol_and_input_integrity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'reports') as directory:
            paths = []
            for seed in range(100, 105):
                rows = [{'index': i, 'words': seed - 98, 'nll': seed / 100.0, 'tokens': 1,
                         'rouge': {k: (seed - 99) / 10 for k in ('rouge1', 'rouge2', 'rougeL')}} for i in range(256)]
                manifest = {'runtime': 'python', 'set': 'development', 'seed': seed, 'data_sha256': DATA_SHA,
                            'protocol_sha256': None, 'source_sha256': {}, 'native_libraries': {},
                            'gpu_identity_and_driver': 'fixture'}
                result = {'schema_version': 1, 'manifest': manifest, 'rows': rows, 'metrics': metrics(rows),
                          'scorer_sha256': 'fixture', 'genppl_sha256': 'fixture', 'model_sha256': {},
                          'versions': {}, 'torch': 'fixture'}
                path = Path(directory) / f'{seed}.json'
                path.write_text(json.dumps(result)); paths.append(path)
            protocol = freeze(paths)
            validate_protocol(protocol)
            self.assertAlmostEqual(protocol['margins']['mean_words'], 3.17)
            protocol['heldout_rows'] = [0, 256]
            with self.assertRaises(ValueError): validate_protocol(protocol)
            bad = json.loads(paths[0].read_bytes()); bad['metrics']['rouge1'] += 1
            paths[0].write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load(paths, 'python', 'development')


if __name__ == '__main__':
    unittest.main()
