import copy
import unittest
from phase1_summary import aggregate, stats


class Phase1Tests(unittest.TestCase):
    def reports(self):
        return [dict(warmups=5, runtime="test",scope="load",source_sha256="fixed",
                     tensor_count=238,byte_exact_readback=True,trials_ms=[float(i+1)]*30) for i in range(3)]

    def test_valid_and_percentiles(self):
        self.assertEqual(aggregate(self.reports(),"load")["samples"],90)
        self.assertEqual(stats([1,2,3])["p95_ms"],3)

    def test_rejects_incomplete_failure_and_changed_source(self):
        for mutation in range(5):
            runs=copy.deepcopy(self.reports())
            if mutation==0: runs.pop()
            if mutation==1: runs[0]["trials_ms"].pop()
            if mutation==2: runs[0]["tensor_count"]=237
            if mutation==3: runs[0]["byte_exact_readback"]=False
            if mutation==4: runs[0]["source_sha256"]="changed"
            with self.assertRaises(ValueError): aggregate(runs,"load")

    def test_nonfinite_and_zero(self):
        for value in (float("nan"),float("inf"),0,-1):
            with self.assertRaises(ValueError): stats([value])


if __name__=="__main__":
    unittest.main()
