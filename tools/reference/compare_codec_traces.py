"""Compare Phase 2 reference captures without changing their declared budgets."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from codec_trace import POLICY, category
from sudoku_artifact import UPSTREAM, digest


def compare(left, right):
    reports = [json.loads((root / "report.json").read_bytes()) for root in (left, right)]
    for root, report in zip((left, right), reports):
        if report["upstream_commit"] != UPSTREAM or report["policy"] != POLICY:
            raise ValueError("reference pin/policy mismatch")
        if digest((root / "traces.safetensors").read_bytes()) != report["trace_sha256"]:
            raise ValueError("trace hash mismatch")
    if reports[0]["config_sha256"] != reports[1]["config_sha256"] or reports[0]["weights"] != reports[1]["weights"]:
        raise ValueError("config or weights differ")
    a = load_file(left / "traces.safetensors")
    b = load_file(right / "traces.safetensors")
    if a.keys() != b.keys():
        raise ValueError("trace inventory differs")
    rows = []
    for name, x in a.items():
        y = b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"trace shape/dtype mismatch: {name}")
        row = {"name": name, "elements": x.numel()}
        if x.is_floating_point():
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("nonfinite trace")
            policy = category(name.split("/", 1)[1])
            budget = POLICY[policy]
            error = (x.double()-y.double()).abs()
            absolute = x.double().abs()
            nonzero = absolute > 0
            row.update(policy=policy, max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
                       max_relative_nonzero=(error[nonzero]/absolute[nonzero]).max().item() if nonzero.any() else None,
                       zero_reference_elements=(~nonzero).sum().item(),
                       mismatches=(error > budget["atol"]+budget["rtol"]*absolute).sum().item())
        else:
            row.update(dtype=str(x.dtype), mismatches=(x != y).sum().item())
        rows.append(row)
    failures = [row for row in rows if row["mismatches"]]
    return {"schema_version": 1, "scope": "reference capture comparison, not Rust model qualification",
            "left_device": reports[0]["device"], "right_device": reports[1]["device"],
            "profile": reports[0]["profile"], "policy": POLICY,
            "left_trace_sha256": reports[0]["trace_sha256"], "right_trace_sha256": reports[1]["trace_sha256"],
            "script_sha256": digest(Path(__file__).read_bytes()),
            "trace_count": len(rows), "all_passed": not failures, "failures": failures, "traces": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path); p.add_argument("right", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--diagnostic", action="store_true", help="record failed comparisons without an error exit; never changes passed status")
    args = p.parse_args()
    torch.set_num_threads(1)
    result = compare(args.left, args.right)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(f"{result['trace_count']} traces, {len(result['failures'])} failing tensors")
    if not result["all_passed"] and not args.diagnostic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
