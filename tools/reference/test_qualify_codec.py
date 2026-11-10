"""Selection qualification tests; fixed float comparisons remain independently gated."""
import unittest
import torch
from qualify_codec import selection


class SelectionTests(unittest.TestCase):
    def test_stable_failure_and_near_tie_remain_visible(self):
        scores = torch.tensor([[0., 100.], [1., 1.000001]])
        result = selection(scores, torch.tensor([0, 0]), torch.tensor([1, 1]), "distance", False)
        self.assertEqual(result["disagreements"], 2)
        self.assertEqual(result["stable_disagreements"], 1)
        self.assertEqual(result["unstable"], 1)
        self.assertEqual(len(result["near_ties_or_disagreements"]), 2)

    def test_reference_first_tie_and_largest(self):
        scores = torch.tensor([[3., 3., 0.], [1., 9., 2.]])
        result = selection(scores, torch.tensor([0, 1]), torch.tensor([0, 1]), "accumulated", True)
        self.assertEqual(result["disagreements"], 0)
        self.assertEqual(result["unstable"], 1)
        with self.assertRaises(ValueError):
            selection(scores, torch.tensor([1, 1]), torch.tensor([1, 1]), "accumulated", True)

    def test_near_tie_does_not_excuse_inconsistent_candidate_selection(self):
        scores = torch.tensor([[1., 1.000001]])
        result = selection(scores, torch.tensor([0]), torch.tensor([1]), "distance", False, scores)
        self.assertEqual(result["stable_disagreements"], 0)
        self.assertEqual(result["candidate_selection_mismatches"], 1)
        with self.assertRaises(ValueError):
            selection(scores, torch.tensor([0]), torch.tensor([2]), "distance", False)

    def test_bad_shape_dtype_and_nonfinite(self):
        with self.assertRaises(ValueError):
            selection(torch.ones(2, 3), torch.zeros(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), "distance", False)
        with self.assertRaises(ValueError):
            selection(torch.ones(2, 3), torch.zeros(2), torch.zeros(2), "distance", False)
        with self.assertRaises(ValueError):
            selection(torch.tensor([[float("nan"), 1.]]), torch.tensor([0]), torch.tensor([0]), "distance", False)


if __name__ == "__main__":
    unittest.main()
