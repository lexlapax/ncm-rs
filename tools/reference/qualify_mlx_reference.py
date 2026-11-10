"""Qualify an adapted Python MLX capture; never relabel it as native or upstream."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from compare_codec_traces import compare
from qualify_codec import selection
from sudoku_artifact import digest


def qualify(reference,candidate):
    reference_bytes=(reference/"report.json").read_bytes()
    anchor=Path(__file__).resolve().parents[2]/"docs/results/2026-11-10-phase2-oracle/raw/sudoku-cuda.json"
    if reference_bytes!=anchor.read_bytes():raise ValueError("untrusted trained CUDA report")
    report=json.loads(reference_bytes)
    actual=json.loads((candidate/"report.json").read_bytes())
    if actual.get("runtime")!="python-mlx-adapted" or actual.get("device")!="metal":raise ValueError("expected adapted Python MLX capture")
    if actual.get("reference_report_sha256")!=digest((reference/"report.json").read_bytes()):raise ValueError("reference identity differs")
    if digest((candidate/"config.json").read_bytes())!=report["config_sha256"]:raise ValueError("candidate config differs")
    if actual.get("uninstrumented_matches_traced") is not True or actual.get("first_index_ties_pass") is not True:raise ValueError("missing execution/tie preflight")
    result=compare(reference,candidate)
    a=load_file(reference/"traces.safetensors");b=load_file(candidate/"traces.safetensors")
    config=json.loads((reference/"config.json").read_bytes());levels=config["vector_quantizer_config"]["levels"]["manual_levels"]
    decisions={}
    for case in report["cases"]:
        name=case["name"];offset=0
        for level,length in enumerate(levels):
            key=f"{name}/encode.level_{level:02}"
            decisions[key]=selection(a[key+".distances"],a[name+"/encode.codes"][:,offset:offset+length].reshape(-1),
                b[name+"/encode.codes"][:,offset:offset+length].reshape(-1),"distance",False,b[key+".distances"])
            offset+=length
        for phase in ("decode","forced"):
            key=name+"/"+phase
            decisions[key]=selection(a[key+".logits"],a[key+".ids"],b[key+".ids"],"accumulated",True,b[key+".logits"])
    passed=result["all_passed"] and all(not r["candidate_selection_mismatches"] for r in decisions.values())
    return {**result,"scope":"independent adapted Python MLX baseline qualification against pinned CUDA, not upstream Apple execution",
        "runtime":"python-mlx-adapted","all_passed":passed,"selections":decisions,"benchmark_baseline_qualified":passed,
        "candidate_report_sha256":digest((candidate/"report.json").read_bytes()),"qualification_script_sha256":digest(Path(__file__).read_bytes())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("reference",type=Path);p.add_argument("candidate",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    torch.set_num_threads(1);result=qualify(a.reference,a.candidate)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(result["trace_count"],"traces; adapted baseline qualified:",result["all_passed"])
    if not result["all_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
