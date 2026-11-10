import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mlx_codec_assets import tensors, save_tensors, load_bundle


class MlxCodecAssetTests(unittest.TestCase):
    def packed(self,entry,payload):
        header=json.dumps({"x":entry}).encode()
        return len(header).to_bytes(8,"little")+header+payload

    def test_roundtrip_without_pickle(self):
        arrays={"x":np.array([[1.,-0.],[2.,3.]],dtype=np.float32),"ids":np.array([0,12],dtype=np.uint32)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"trace.safetensors";save_tensors(path,arrays)
            result=tensors(path.read_bytes())
            self.assertEqual(result["x"].tobytes(),arrays["x"].tobytes())
            self.assertEqual(result["ids"].tolist(),[0,12]);self.assertEqual(result["ids"].dtype,np.int64)
            with self.assertRaises(FileExistsError):save_tensors(path,arrays)

    def test_rejects_malformed_layouts(self):
        for raw in (b"",(100).to_bytes(8,"little"),
                    self.packed({"dtype":"F32","shape":[2],"data_offsets":[0,4]},b"\0"*4),
                    self.packed({"dtype":"F32","shape":[1],"data_offsets":[1,5]},b"\0"*5),
                    self.packed({"dtype":"F32","shape":[0],"data_offsets":[0,0]},b""),
                    self.packed({"dtype":"F16","shape":[1],"data_offsets":[0,2]},b"\0"*2)):
            with self.assertRaises(ValueError):tensors(raw)

    def test_rejects_unpinned_manifest_before_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"pinned Sudoku"):load_bundle(root)


if __name__=="__main__":unittest.main()
