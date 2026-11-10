"""Apple-only matched summaries with an explicitly qualified adapted baseline."""
import argparse
import hashlib
import json
from pathlib import Path
from phase2_summary import ROOT, require, reports, codec, tokenizers


def check_baseline(root):
    # Benchmark runs stay separate from the fixed, independently-qualified
    # baseline evidence. Do not permit a copied or substituted baseline beside
    # the mutable measurement directory.
    baseline_root=ROOT/"reports/phase2-codec/python-mlx-v1"
    capture_raw=(baseline_root/"report.json").read_bytes();capture=json.loads(capture_raw)
    quality_raw=(ROOT/"reports/phase2-codec/python-mlx-v1-qualification.json").read_bytes();quality=json.loads(quality_raw)
    oracle_raw=(ROOT/"docs/results/2026-11-10-phase2-oracle/raw/sudoku-cuda.json").read_bytes();oracle=json.loads(oracle_raw)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    require(capture["runtime"]=="python-mlx-adapted" and capture["device"]=="metal","incorrect baseline runtime")
    require(capture["reference_report_sha256"]==sha(oracle_raw) and capture["policy"]==oracle["policy"],"incorrect baseline oracle/policy")
    require(quality["candidate_report_sha256"]==sha(capture_raw) and quality["left_trace_sha256"]==oracle["trace_sha256"] and quality["right_trace_sha256"]==capture["trace_sha256"],"baseline identity mismatch")
    require(capture["trace_count"]==quality["trace_count"]==642 and len(quality["traces"])==642,"incomplete baseline")
    require(quality["all_passed"] is True and quality["benchmark_baseline_qualified"] is True and not quality["failures"] and all(r["mismatches"]==0 for r in quality["traces"]),"failed baseline")
    require(sum(r["decisions"] for r in quality["selections"].values())==2500 and all(r["disagreements"]==r["candidate_selection_mismatches"]==0 for r in quality["selections"].values()),"baseline selection failure")
    runs=reports(root,"codec")
    for runtime,group in runs.items():
        for run in group:
            require(run["runtime"]==("rust-mlx" if runtime=="rust" else "python-mlx-adapted"),"runtime mismatch")
            if runtime=="python":
                require(run["baseline_qualification_sha256"]==sha(quality_raw),"unqualified benchmark reference")
                for name in ("tools/reference/mlx_codec.py","tools/reference/mlx_codec_assets.py","tools/reference/mlx_operators.py","tools/reference/requirements-macos.lock"):
                    require(run["source_sha256"][name]==capture["source_sha256"][name],"baseline code changed before measurement")
    return sha(quality_raw)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    baseline=check_baseline(a.directory)
    result={"schema_version":1,"device":"metal","baseline_qualification_sha256":baseline,"codec":codec(a.directory,"metal"),"tokenizers":tokenizers(a.directory),
        "all_benchmark_output_checks_passed":True,"source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("tools/bench/phase2_mlx_summary.py","tools/bench/phase2_summary.py")},
        "limitations":["Same-device adapted Python MLX port, not unmodified upstream Apple execution or a pure language speedup claim.",
            "Stream/kernel-only timing is unavailable and represented as null. Resident middle-block timing excludes host transfers but includes graph construction/evaluation.",
            "Separate stage barriers perturb execution; final selection includes host readback, unlike CUDA stream-event spans.",
            "MLX allocator active/peak/cache is unified memory, not separate VRAM or whole-process RSS. Descriptive three-process statistics, not independent 90-process confidence bounds."]}
    with a.output.open("x") as file:json.dump(result,file,indent=2,allow_nan=False);file.write("\n")


if __name__=="__main__":main()
