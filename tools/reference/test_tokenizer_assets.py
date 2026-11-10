"""Data-only conversion rejection tests; no downloaded assets or unrestricted load."""
import pickle
import struct
import unittest
from unittest import mock
from convert_tokenizers import data_pickle, validate_bpe


class DataOnlyTests(unittest.TestCase):
    def test_scalar_container_data_matches_without_unpickling(self):
        for value in [{0:b"a",1:b"\xff"},[(b"a",b"b"),(b"a",b"b")]]:
            data = pickle.dumps(value,protocol=4)
            with mock.patch("pickle.loads",side_effect=AssertionError("must not unpickle")):
                self.assertEqual(data_pickle(data),value)

    def test_executable_opcodes_and_metadata_are_rejected(self):
        for value in [eval, {"metadata":"unsupported"}, {0:True}, {0:bytearray(b"a")}]:
            with self.assertRaises(ValueError):
                data_pickle(pickle.dumps(value,protocol=4))

    def test_framing_protocol_trailing_and_bounds(self):
        data = pickle.dumps({0:b"a",1:b"b"},protocol=4)
        for invalid in [data+b"trailing", data[:3], data[:3]+struct.pack("<Q",2**63)+data[11:],
                        pickle.dumps({0:b"a"},protocol=3), b"x"*(2*1024*1024+1)]:
            with self.assertRaises((ValueError,IndexError)):
                data_pickle(invalid)

    def test_duplicate_keys_do_not_silently_overwrite(self):
        data = pickle.dumps({0:b"a",1:b"b"},protocol=4)
        self.assertEqual(data.count(b"K\x01"),1)
        with self.assertRaises(ValueError):
            data_pickle(data.replace(b"K\x01",b"K\x00"))

    def test_schema_rejects_wrong_inventory_and_recursive_data(self):
        with self.assertRaises(ValueError): validate_bpe({0:b"a"},[])
        recursive=[]; recursive.append(recursive)
        with self.assertRaises(ValueError): validate_bpe({i:recursive for i in range(8192)},[])
        with self.assertRaises(ValueError): validate_bpe({i:b"same" for i in range(8192)},[])


if __name__ == "__main__":
    unittest.main()
