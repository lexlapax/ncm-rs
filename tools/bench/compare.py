"""Summarize three fresh-process CUDA operator probes per runtime; no model claims."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(reports):
    if len(reports) != 6:
        raise ValueError("expected three Python and three Rust reports")
    shared = ("fixture_sha256", "upstream_commit", "device", "dtype", "tf32", "scope", "atol", "rtol")
    reference = reports[0]
    grouped = {}
    for report in reports:
        if any(report[key] != reference[key] for key in shared):
            raise ValueError("incompatible report provenance/settings")
        if report["device"] != "cuda" or report["dtype"] != "float32" or report["tf32"]:
            raise ValueError("this comparison requires FP32 CUDA without TF32")
        if not report["passed"] or report["warmups"] != 5 or report["trials"] != 30:
            raise ValueError("failed or incomplete probe")
        if len({case["name"] for case in report["results"]}) != len(report["results"]):
            raise ValueError("duplicate operator name")
        for case in report["results"]:
            if not case["errors"]["passed"] or case["errors"]["mismatches"]:
                raise ValueError("numerical regression")
            times = case["trials_ms"]
            if len(times) != 30 or any(not math.isfinite(x) or x <= 0 for x in times):
                raise ValueError("invalid timing trials")
        grouped.setdefault(report["runtime"], []).append(report)
    if set(grouped) != {"candle-0.11.1", "pytorch-eager-math-sdpa"}:
        raise ValueError("unexpected runtimes")
    names = [case["name"] for case in reference["results"]]
    if not names:
        raise ValueError("empty operator suite")
    aggregates = {}
    for runtime, runs in grouped.items():
        if len(runs) != 3:
            raise ValueError("expected three fresh-process runs per runtime")
        for identity in ("compiled_source_sha256", "script_sha256", "torch_version", "cuda_version"):
            if any(run.get(identity) != runs[0].get(identity) for run in runs):
                raise ValueError("source/library changed between trials")
        if any([case["name"] for case in run["results"]] != names for run in runs):
            raise ValueError("operator suites differ")
        values = []
        for index, name in enumerate(names):
            cases = [run["results"][index] for run in runs]
            times = sorted(t for case in cases for t in case["trials_ms"])
            medians = [statistics.median(case["trials_ms"]) for case in cases]
            values.append({
                "name": name, "median_ms": statistics.median(times),
                "p95_ms": times[math.ceil(0.95 * len(times)) - 1],
                "process_median_range_ms": [min(medians), max(medians)],
                "max_abs_error": max(case["errors"]["max_abs_error"] for case in cases),
                "mismatches": sum(case["errors"]["mismatches"] for case in cases),
                "trials": len(times),
            })
        aggregates[runtime] = values
    return {"schema_version": 1, "status": "operator-probe-only",
            **{key: reference[key] for key in shared}, "runs_per_runtime": 3,
            "quality": {"operator_parity": "passed", "model_quality": None},
            "peak_memory": None, "runtimes": aggregates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    summary = summarize([json.loads(path.read_text()) for path in args.reports])
    summary["raw_reports_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.reports
    }
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
