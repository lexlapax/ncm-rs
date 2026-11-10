"""Offline exporter rejection tests; requires the pinned reference venv, no weights/GPU."""
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from sudoku_artifact import checkpoint, export, tensor_bytes


def serialized(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


class ExportTests(unittest.TestCase):
    def test_weights_only_and_bytes(self):
        data = serialized({"model": {"weight": torch.tensor([0.0, -0.0, float("-inf")])}})
        real_load = torch.load
        with patch("sudoku_artifact.torch.load", wraps=real_load) as load:
            state = checkpoint(data)
        self.assertIs(load.call_args.kwargs["weights_only"], True)
        self.assertEqual(len(tensor_bytes(state["weight"])), 12)
        self.assertEqual(tensor_bytes(torch.tensor(7, dtype=torch.int64)), (7).to_bytes(8, "little"))

    def test_rejects_metadata_and_unsupported_types(self):
        for value in ({"model": {}, "optimizer": {}}, {"model": {"x": 7}},
                      {"model": {"x": torch.ones(2, dtype=torch.float64)}}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint(serialized(value))

    def test_rejects_shared_storage_including_different_offsets(self):
        base = torch.arange(4, dtype=torch.float32)
        for second in (base[:2], base[2:]):
            with self.assertRaisesRegex(ValueError, "shared storage"):
                checkpoint(serialized({"model": {"a": base[:2], "b": second}}))

    def test_no_unrestricted_retry(self):
        with patch("sudoku_artifact.torch.load", side_effect=ValueError("unsafe")) as load:
            with self.assertRaises(ValueError):
                checkpoint(b"bad")
            self.assertEqual(load.call_count, 1)
            self.assertIs(load.call_args.kwargs["weights_only"], True)

    def test_bad_input_never_loads_and_existing_output_refused(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.yaml").write_text("bad")
            with patch("sudoku_artifact.torch.load") as load:
                with self.assertRaises(ValueError):
                    export(root, root / "output")
                load.assert_not_called()
            self.assertFalse((root / "output").exists())
            with self.assertRaises(FileExistsError):
                export(root, root)


if __name__ == "__main__":
    unittest.main()
