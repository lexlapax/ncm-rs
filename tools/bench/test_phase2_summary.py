import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from phase2_summary import ROOT, SCOPES, aggregate, codec, floats, stats, tokenizers


class Phase2SummaryTests(unittest.TestCase):
    def test_statistics_and_trial_contract(self):
        rows=[{"v":[float(i+1)]*30} for i in range(3)]
        self.assertEqual(aggregate(rows,"v")["n"],90)
        self.assertEqual(aggregate(rows,"v")["process_medians_ms"],[1,2,3])
        self.assertEqual(stats([1,2,3])["p95_ms"],3)
        for value in (0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):stats([value])
        with self.assertRaises(ValueError):aggregate(rows[:2],"v")
        rows[0]["v"].pop()
        with self.assertRaises(ValueError):aggregate(rows,"v")

    def test_fixed_float_budget(self):
        self.assertEqual(floats([0,1],[0.0005,1])["elements"],2)
        for candidate in ([],[0],[0,float("nan")],[0,float("inf")],[0.0005001,1]):
            with self.assertRaises(ValueError):floats([0,1],candidate)

    def test_tokenizer_matrix_and_output_rejection(self):
        raw=(ROOT/"tests/fixtures/tokenizers-v1.json").read_bytes();fixture=json.loads(raw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for profile,record in fixture["profiles"].items():
                report={"schema_version":1,"warmups":5,"trials":30,"source_sha256":{"test":"fixed"},
                    "profile":profile,"fixture_sha256":hashlib.sha256(raw).hexdigest(),"manifest_sha256":record["manifest_sha256"],
                    "strings_per_trial":89,"input_utf8_bytes":sum(len(c["text"].encode()) for c in record["cases"]),
                    "token_positions":sum(len(c["ids"]) for c in record["cases"]),"load_trials_ms":[1.]*30,"first_load_ms":1.,"host_final":{},
                    "measurements":[{"scope":scope,"trials_ms":[1.]*30,"first_ms":1.,"output_sha256":"fixed"} for scope in ("encode","decode","decode_skip_special")]}
                for runtime in ("rust","python"):
                    for i in range(1,4):(root/f"tokenizer-{profile}-{runtime}-{i}.json").write_text(json.dumps(report))
            self.assertEqual([r["non_special_token_positions"] for r in tokenizers(root)],[1261,2274,1974,2982])
            path=root/"tokenizer-bert-rust-1.json";report=json.loads(path.read_bytes());report["measurements"][0]["output_sha256"]="wrong";path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):tokenizers(root)

    def fixture(self):
        sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        report={"schema_version":1,"warmups":5,"trials":30,"source_sha256":{"test":"fixed"},
            "manifest_sha256":sha("docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"),
            "workload_sha256":sha("tests/fixtures/codec-bench-v1.json"),"precision":"fp32","device":"cuda",
            "load_trials_ms":[1.]*30,"first_load_ms":1.,"first_execution_ms":1.,"load_memory":{},"host_final":{"VmHWM":1},
            "measurements":[],"instrumented_stages":[]}
        for batch in (1,2,4):
            report["instrumented_stages"].append({"batch":batch,"cuda_stream_spans_ms":{name:[1.]*30 for name in ["encoder_pre_quant","decoder_logits","output_selection"]+[f"encode.level_{i:02}" for i in range(12)]}})
            for scope in SCOPES:
                count=batch*{"encode":322,"decode":89,"reconstruct":411,"resident_decoder_block":0}[scope]
                retained=batch*89*{"encode":12*256,"decode":12*256+12,"reconstruct":24*256+12,"resident_decoder_block":256}[scope]
                report["measurements"].append({"batch":batch,"scope":scope,"input_ids_sha256":"fixed","supplied_codes_sha256":"fixed","output_ids_sha256":"fixed",
                    "output_ids_count":count,"output_float_elements":retained,"first_logits":[0.]*(batch*89*12) if scope in ("decode","reconstruct") else None,
                    "trials_ms":[1.]*30,"cuda_stream_spans_ms":[1.]*30,"first_ms":1.,"host":{"VmHWM":1},"pool_bytes":{},"allocator_bytes":{}})
        return report

    def test_full_matrix_and_reject_mutations(self):
        baseline=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for runtime in ("rust","python"):
                for i in range(1,4):(root/f"codec-{runtime}-{i}.json").write_text(json.dumps(baseline))
            self.assertEqual(len(codec(root)["measurements"]),12)
            mutations=[lambda r:r["measurements"].pop(),lambda r:r["measurements"].append(r["measurements"][0]),
                lambda r:r["source_sha256"].update(test="changed"),lambda r:r.update(manifest_sha256="wrong"),
                lambda r:r["measurements"][0].update(output_ids_sha256="wrong"),lambda r:r["measurements"][0].update(output_float_elements=0),
                lambda r:r["measurements"][1].update(first_logits=[]),lambda r:r["instrumented_stages"][0]["cuda_stream_spans_ms"].pop("encoder_pre_quant"),
                lambda r:r["measurements"][1]["first_logits"].__setitem__(0,.001)]
            for mutate in mutations:
                report=copy.deepcopy(baseline);mutate(report)
                (root/"codec-rust-1.json").write_text(json.dumps(report))
                with self.assertRaises(ValueError):codec(root)


if __name__=="__main__":unittest.main()
