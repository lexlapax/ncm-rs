"""Aggregate the three-process Apple Phase 1 probes; never infer model quality."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def stats(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("nonpositive/nonfinite timing")
    return {"median_ms":statistics.median(samples),"p95_ms":sorted(samples)[math.ceil(.95*len(samples))-1],
            "min_ms":min(samples),"max_ms":max(samples),"samples":len(samples)}


def aggregate(runs, kind):
    if len(runs)!=3:
        raise ValueError("three fresh-process reports required")
    for r in runs:
        if r.get("warmups")!=5:
            raise ValueError("warmup mismatch")
        for key in ("runtime","scope","source_sha256","fixture_sha256","manifest_sha256"):
            if r.get(key)!=runs[0].get(key):
                raise ValueError(f"inconsistent {key}")
    if kind=="operators":
        names=[x["name"] for x in runs[0]["results"]]
        if not names or len(names)!=len(set(names)):
            raise ValueError("empty/duplicate cases")
        for r in runs:
            if not r.get("passed") or [x["name"] for x in r["results"]]!=names:
                raise ValueError("operator failure or inventory mismatch")
        result={}
        for i,name in enumerate(names):
            cases=[r["results"][i] for r in runs]
            if any(not c["errors"]["passed"] or c["errors"]["mismatches"]!=0 for c in cases):
                raise ValueError("numerical failure")
            for c in cases:
                if len(c["trials_ms"])!=30:
                    raise ValueError("trial count mismatch")
            result[name]=stats([x for c in cases for x in c["trials_ms"]])
            result[name]["max_abs_error"]=max(c["errors"]["max_abs_error"] for c in cases)
        return result
    for r in runs:
        if len(r["trials_ms"])!=30:
            raise ValueError("trial count mismatch")
        if kind=="load" and (r.get("tensor_count")!=238 or not r.get("byte_exact_readback")):
            raise ValueError("tensor readback incomplete")
        if kind=="tokenizer" and (not r.get("parity_passed") or r.get("cases")!=5 or r.get("repetitions")!=1024):
            raise ValueError("tokenizer parity/workload mismatch")
    out=stats([x for r in runs for x in r["trials_ms"]])
    out["per_process_median_ms"]=[statistics.median(r["trials_ms"]) for r in runs]
    return out


def summarize(root):
    output={}
    for kind in ("operators","load","tokenizer"):
        reports={runtime:[json.loads((root/f"{runtime}-{kind}-{i}.json").read_bytes()) for i in (1,2,3)] for runtime in ("rust","python")}
        for key in ("scope","fixture_sha256","manifest_sha256"):
            if reports["rust"][0].get(key)!=reports["python"][0].get(key):
                raise ValueError(f"cross-runtime mismatch: {key}")
        output[kind]={runtime:aggregate(runs,kind) for runtime,runs in reports.items()}
    return {"schema_version":1,"processes_per_runtime":3,"results":output,"model_quality":None,
            "aggregator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path); p.add_argument("output",type=Path)
    args=p.parse_args()
    result=summarize(args.root)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n")


if __name__=="__main__":
    main()
