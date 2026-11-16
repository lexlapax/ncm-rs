"""Bootstrap must preserve frozen directions and detect a clear regression."""
import unittest
from xsum_mlx_stochastic_analyze import bootstrap


def rows(rouge, nll, words):
    return [{'rows': [{'rouge': {'rouge1': rouge, 'rouge2': rouge, 'rougeL': rouge},
                       'nll': nll, 'tokens': 1, 'words': words} for _ in range(8)]} for _ in range(5)]


class MetalBootstrapTests(unittest.TestCase):
    def test_equal_inputs_and_injected_degradation(self):
        reference = rows(0.5, 1.0, 10)
        observed, bounds = bootstrap(reference, reference, replicates=100)
        for key in observed:
            self.assertAlmostEqual(observed[key], 0.0)
            self.assertAlmostEqual(bounds[key], 0.0)
        degraded = rows(0.4, 1.2, 13)
        observed, bounds = bootstrap(reference, degraded, replicates=100)
        self.assertAlmostEqual(observed['rouge1'], 10.0)
        self.assertGreater(bounds['generation_perplexity'], 0)
        self.assertAlmostEqual(bounds['mean_words'], 3.0)

    def test_rejects_unequal_row_counts(self):
        with self.assertRaises(ValueError):
            bootstrap(rows(0.5, 1.0, 10), rows(0.5, 1.0, 10)[:4], replicates=1)


if __name__ == '__main__':
    unittest.main()
