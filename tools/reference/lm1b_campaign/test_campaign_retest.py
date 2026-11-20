"""Offline controls for the explicit nonblind LM1B CUDA retest mode."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
# The policy controls need no Torch or weights. Stub the unrelated exporter import.
stub = types.ModuleType('tinystories_artifact')
stub.ROOT = HERE.parents[2]
stub.UPSTREAM = 'stub'
stub.verify_input = lambda *args: None
with patch.dict(sys.modules, {'tinystories_artifact': stub}):
    campaign = importlib.import_module('campaign')


def frozen():
    return {'python': list(range(1000, 1080)), 'rust': list(range(2000, 2080))}


class RetestPolicyTests(unittest.TestCase):
    def test_retest_requires_exact_frozen_ordered_cuda_cohorts(self):
        self.assertEqual(campaign.seed_lists('retest', frozen()), frozen())
        for mutated in (
            {'python': list(range(1001, 1081)), 'rust': list(range(2000, 2080))},
            {'python': list(reversed(range(1000, 1080))), 'rust': list(range(2000, 2080))},
            {'python': list(range(1000, 1080)), 'rust': list(range(1000, 1080))},
        ):
            with self.subTest(mutated=mutated), self.assertRaises(ValueError):
                campaign.seed_lists('retest', mutated)

    def test_retest_manifest_requires_original_registry_identity(self):
        plan = {'schema_version': 1, 'params': campaign.PARAMS, 'mode': 'retest',
                'seeds': frozen(), 'seed_registry': {'sha256': '0' * 64}, 'pins': {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(plan))
            with patch.object(campaign, 'pins', return_value={}):
                with self.assertRaisesRegex(ValueError, 'registry identity'):
                    campaign.manifest(path, None)
                plan['seed_registry']['sha256'] = campaign.RETEST_REGISTRY_SHA
                path.write_text(json.dumps(plan))
                self.assertEqual(campaign.manifest(path, None)['mode'], 'retest')

    def test_retest_fails_closed_when_metrics_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.object(campaign, 'read_scores', return_value=[]), \
                    patch.object(campaign, 'compare_cohorts', return_value={'metrics_passed': False}):
                self.assertEqual(campaign.summary({'mode': 'retest', 'seeds': frozen()}, 'abc', run), 1)


if __name__ == '__main__':
    unittest.main()
