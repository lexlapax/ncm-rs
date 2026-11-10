"""Compare finite PoC outputs with pinned Torch CPU interpolation, without timing."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2.15.0+cu132"
COMMIT = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
LENGTHS = (1, 2, 3, 4, 6, 7, 9, 13, 16, 18, 26, 37, 50, 67, 89)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    if torch.__version__ != VERSION or torch.version.git_version != COMMIT:
        raise ValueError("unqualified Torch version; do not silently change the reference")
    torch.set_num_threads(1)
    raw = args.report.read_bytes()
    report = json.loads(raw)
    if report["schema_version"] != 1 or report["precision"] != "fp32" or report["device"] not in ("cpu", "cuda"):
        raise ValueError("unsupported PoC report")
    # Use a list rather than a set: 89->89 occurs in both directions deliberately.
    expected = [(b, i, o, 16, s, k) for b in (1, 2, 4) for length in LENGTHS
                for i, o in ((length, 89), (89, length)) for s in (False, True)
                for k in ("linear", "area")]
    actual = [tuple(row[key] for key in ("batch", "input_length", "output_length", "channels", "strided", "kind"))
              for row in report["rows"]]
    if actual != expected:
        raise ValueError("incomplete/reordered PoC matrix")
    if report.get("coordinate_math") not in ("f32", "f64"):
        raise ValueError("missing coordinate mode")
    comparisons = []
    for row in report["rows"]:
        b, i, o, c = (row[k] for k in ("batch", "input_length", "output_length", "channels"))
        x = ((torch.arange(b*i*c) * 73 % 1021).float() - 510).reshape(b, i, c) / 64
        if row["strided"]:
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
        options = {"align_corners": False} if row["kind"] == "linear" else {}
        ref = F.interpolate(x.transpose(1, 2), size=o, mode=row["kind"], **options).transpose(1, 2).reshape(-1)
        candidate = torch.tensor(row["output"], dtype=torch.float32)
        if ref.shape != candidate.shape or not torch.isfinite(candidate).all():
            raise ValueError("nonfinite or wrong-sized candidate")
        delta = (candidate-ref).abs()
        budget = 1e-5 + 1e-4 * ref.abs()
        mismatches = int((delta > budget).sum())
        comparisons.append({k:v for k,v in row.items() if k != "output"} | {
            "elements": ref.numel(), "mismatches": mismatches,
            "max_abs_vs_torch": float(delta.max()), "max_budget_fraction": float((delta/budget).max())})
    result = {"schema_version": 1, "scope": "PoC vs Torch CPU, not trained GPU qualification or performance",
              "coordinate_math": report["coordinate_math"],
              "report_sha256": hashlib.sha256(raw).hexdigest(), "candidate_source_sha256": report["source_sha256"],
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_version": VERSION, "torch_commit": COMMIT,
              "all_passed": all(r["mismatches"] == 0 for r in comparisons), "rows": comparisons}
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(len(comparisons), "rows; all passed:", result["all_passed"])
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
