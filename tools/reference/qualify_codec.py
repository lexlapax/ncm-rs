"""Qualify native codec traces against the fixed Phase 2 reference contract."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from codec_trace import POLICY, category, margins
from compare_codec_traces import compare
from sudoku_artifact import digest


def selection(scores, expected, actual, policy, largest, actual_scores=None):
    if expected.shape != scores.shape[:-1] or actual.shape != expected.shape:
        raise ValueError("selection shape differs")
    if expected.dtype != torch.int64 or actual.dtype != torch.int64:
        raise ValueError("selection must contain checked I64 IDs")
    if ((actual < 0) | (actual >= scores.shape[-1])).any():
        raise ValueError("candidate selection outside score axis")
    candidate_mismatches = 0
    if actual_scores is not None:
        if actual_scores.shape != scores.shape or not torch.isfinite(actual_scores).all():
            raise ValueError("invalid candidate scores")
        selected = actual_scores.argmax(-1) if largest else actual_scores.argmin(-1)
        candidate_mismatches = (selected != actual).sum().item()
    oracle = scores.argmax(-1) if largest else scores.argmin(-1)
    if not torch.equal(oracle, expected):
        raise ValueError("reference IDs do not match first-index selection")
    margin = margins(scores, policy, largest)
    stable = torch.tensor(margin["stable"], dtype=torch.bool).reshape(expected.shape)
    disagreement = actual != expected
    details = []
    for position in ((~stable) | disagreement).nonzero().tolist():
        index = tuple(position)
        flat = 0
        for coordinate, size in zip(position, expected.shape):
            flat = flat * size + coordinate
        details.append({"position": position, "reference": expected[index].item(),
                        "actual": actual[index].item(), "stable": stable[index].item(),
                        "gap": margin["gaps"][flat]})
    return {"decisions": expected.numel(), "disagreements": disagreement.sum().item(),
            "stable_disagreements": (stable & disagreement).sum().item(),
            "candidate_selection_mismatches": candidate_mismatches,
            "unstable": (~stable).sum().item(), "minimum_gap": margin["minimum_gap"],
            "near_ties_or_disagreements": details}


def qualify(reference, candidate):
    comparison = compare(reference, candidate)
    ref_report = json.loads((reference / "report.json").read_bytes())
    actual_report = json.loads((candidate / "report.json").read_bytes())
    if actual_report.get("runtime") not in ("rust-candle", "rust-mlx"):
        raise ValueError("candidate must identify a native Rust runtime")
    a = load_file(reference / "traces.safetensors")
    b = load_file(candidate / "traces.safetensors")
    config_bytes = (reference / "config.json").read_bytes()
    if digest(config_bytes) != ref_report["config_sha256"]:
        raise ValueError("reference config hash mismatch")
    if actual_report.get("reference_report_sha256") != digest((reference / "report.json").read_bytes()):
        raise ValueError("candidate reference identity differs")
    if not a or not ref_report.get("cases") or actual_report["profile"] != ref_report["profile"]:
        raise ValueError("empty or inconsistent qualification profile")
    config = json.loads(config_bytes)
    levels = config["vector_quantizer_config"]["levels"]["manual_levels"]
    decisions, reconstruction = {}, {}
    for case in ref_report["cases"]:
        name = case["name"]
        codes = a[name+"/encode.codes"]
        actual_codes = b[name+"/encode.codes"]
        offset = 0
        for level, length in enumerate(levels):
            key = f"{name}/encode.level_{level:02}"
            scores = a[key+".distances"]
            decisions[key] = selection(scores, codes[:, offset:offset+length].reshape(-1),
                                       actual_codes[:, offset:offset+length].reshape(-1), "distance", False, b[key+".distances"])
            offset += length
        if offset != codes.shape[1]:
            raise ValueError("code level inventory differs")
        for phase in ("decode", "forced"):
            key = name+"/"+phase
            expected, actual = a[key+".ids"], b[key+".ids"]
            decisions[key] = selection(a[key+".logits"], expected, actual, "accumulated", True, b[key+".logits"])
            original = a[name+"/input.ids"]
            reconstruction[key] = {
                "tokens": expected.numel(), "sequences": expected.shape[0],
                "reference_agreement_tokens": (expected == actual).sum().item(),
                "reference_agreement_sequences": (expected == actual).all(-1).sum().item(),
                "reference_recovers_input_tokens": (expected == original).sum().item(),
                "native_recovers_input_tokens": (actual == original).sum().item(),
                "note": "lossy codec reconstruction, not Sudoku generation/solving quality"}
    # Keep every integer disagreement visible. Only stable output/code IDs gate;
    # fixed input IDs always gate, and near ties never excuse floating failures.
    floating_failures = [r for r in comparison["failures"] if "policy" in r]
    input_failures = [r for r in comparison["failures"] if "policy" not in r and not r["name"].endswith(("encode.codes", "decode.ids", "forced.ids"))]
    stable_failures = sum(r["stable_disagreements"] + r["candidate_selection_mismatches"] for r in decisions.values())
    edges = [0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    histograms = {key: {"elements": 0, "absolute_error_sum": 0., "max_budget_fraction": 0., "bins": [0]*(len(edges)+1)} for key in POLICY}
    for name, expected in a.items():
        if not expected.is_floating_point():
            continue
        error = (expected.double()-b[name].double()).abs().flatten()
        key = category(name.split("/", 1)[1])
        histogram = histograms[key]
        histogram["elements"] += error.numel()
        histogram["absolute_error_sum"] += error.sum().item()
        budgets = POLICY[key]["atol"] + POLICY[key]["rtol"] * expected.double().abs().flatten()
        fractions = error / budgets
        index = fractions.argmax().item()
        fraction = fractions[index].item()
        if fraction > histogram["max_budget_fraction"]:
            histogram["max_budget_fraction"] = fraction
            histogram["worst_budget_fraction"] = {"name": name, "flat_index": index,
                "reference": expected.flatten()[index].item(), "actual": b[name].flatten()[index].item(),
                "absolute_error": error[index].item(), "allowed_error": budgets[index].item()}
        counts = torch.bincount(torch.bucketize(error, torch.tensor(edges, dtype=torch.float64)), minlength=len(edges)+1)
        histogram["bins"] = [x+y for x, y in zip(histogram["bins"], counts.tolist())]
    return {**comparison, "scope": "native Rust codec trace qualification; no performance claim",
            "runtime": actual_report["runtime"], "candidate_report_sha256": digest((candidate / "report.json").read_bytes()),
            "all_trace_elements_pass": comparison["all_passed"],
            "all_passed": not floating_failures and not input_failures and not stable_failures,
            "qualification_script_sha256": digest(Path(__file__).read_bytes()),
            "selections": decisions, "reconstruction": reconstruction,
            "absolute_error_histogram_upper_inclusive_edges": edges,
            "absolute_error_histograms": histograms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path); parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = qualify(args.reference, args.candidate)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False); file.write("\n")
    print(f"{result['trace_count']} traces; passed={result['all_passed']}; "
          f"ID disagreements={sum(x['disagreements'] for x in result['selections'].values())}; "
          f"near ties={sum(x['unstable'] for x in result['selections'].values())}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
