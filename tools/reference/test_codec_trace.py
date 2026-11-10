"""Reference-contract unit tests (pinned offline venv; no weights or GPU needed)."""
import unittest
import torch
from codec_trace import POLICY, category, margins, reference_inputs


class CodecContractTests(unittest.TestCase):
    def test_synthetic_input_shapes_and_sudoku_separators(self):
        cases = reference_inputs(89)
        self.assertEqual({len(batch) for batch in cases.values()}, {1, 2})
        for batch in cases.values():
            for row in batch:
                self.assertEqual(len(row), 89)
                self.assertEqual([i for i, token in enumerate(row) if token == 10], list(range(9, 89, 10)))
                self.assertTrue(all(0 <= token < 12 for token in row))
        self.assertEqual(len(reference_inputs(16)["pattern_batch2"][0]), 16)

    def test_policy_boundary_classification(self):
        self.assertEqual(category("encode.pre_quant"), "component")
        self.assertEqual(category("encode.level_00.distances"), "distance")
        self.assertEqual(category("decode.level_00.cumulative"), "accumulated")
        self.assertEqual(category("decode.decoder.middle_block.output"), "accumulated")
        self.assertEqual(POLICY["primitive"], {"atol": 1e-5, "rtol": 1e-4})

    def test_stable_and_tied_score_rows(self):
        values = torch.tensor([[1., 2., 3.], [1., 1., 3.]])
        result = margins(values, "distance", largest=False)
        self.assertEqual(result["stable"], [True, False])
        self.assertEqual(result["gaps"], [1., 0.])
        self.assertEqual(values.argmin(-1).tolist(), [0, 0])
        self.assertEqual(margins(-values, "accumulated", largest=True)["stable"], [True, False])

    def test_rejects_invalid_score_rows(self):
        for values in (torch.tensor(1.), torch.zeros(1, 1), torch.zeros(0, 2), torch.tensor([[1., float("nan")]])):
            with self.assertRaises(ValueError):
                margins(values, "distance", largest=False)


if __name__ == "__main__":
    unittest.main()
