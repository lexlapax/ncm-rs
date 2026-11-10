"""Descriptive paired-process uncertainty for the fixed TinyStories control run.

Consumes every process declared in environment.json. Does not close a phase gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from phase2_summary import ROOT, require, stats


def interval(candidate, comparator, rng, repeats=10000):
    require(len(candidate) == len(comparator) >= 3, "unpaired processes")
    samples = [[], []]
    for _ in range(repeats):
        selected = rng.choices(range(len(candidate)), k=len(candidate))
        a, b = (sorted(v for i in selected for v in rows[i]) for rows in (candidate, comparator))
        samples[0].append(statistics.median(a)/statistics.median(b))
        samples[1].append(a[math.ceil(.95*len(a))-1]/b[math.ceil(.95*len(b))-1])
    return {name: [sorted(values)[math.ceil(.025*repeats)-1], sorted(values)[math.ceil(.975*repeats)-1]]
            for name, values in zip(("median_ratio", "p95_ratio"), samples)}


def summarize(root, profile="tinystories", include_load=False):
    environment = json.loads((root/"environment.json").read_bytes())
    apple_control = (environment.get("platform") == "Darwin"
                     and environment.get("control_policy") == "counterbalanced_fresh_processes"
                     and environment.get("cpu_affinity") is None)
    require(profile in ("bert", "gpt2", "qwen3", "tinystories") and profile in environment["workloads"]
            and (environment["cpu_affinity"] is not None or apple_control),
            "not a declared controlled tokenizer run")
    if apple_control:
        require(not environment.get("telemetry_error") and (root/"cpu-activity.jsonl").is_file()
                and (root/"cpu-activity.jsonl").stat().st_size > 0, "missing Apple activity telemetry")
    count = environment["repeats"]
    require(count >= 3, "insufficient processes")
    fixture = (ROOT/"tests/fixtures/tokenizers-v1.json").read_bytes()
    manifest = json.loads(fixture)["profiles"][profile]["manifest_sha256"]
    groups = {}
    fingerprints = {}
    for name in ("previous", "candidate", "python"):
        suffix = "python" if name == "python" else "rust"
        paths = [root/name/f"tokenizer-{profile}-{suffix}-{i}.json" for i in range(1,count+1)]
        require(set((root/name).glob(f"tokenizer-{profile}-{suffix}-*.json")) == set(paths), "missing/extra processes")
        groups[name] = [json.loads(p.read_bytes()) for p in paths]
        fingerprints[name] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        for run in groups[name]:
            require(run["schema_version"] == 1 and run["warmups"] == 5 and run["trials"] == 30, "protocol mismatch")
            require(run["runtime"] == ("pinned-python-tokenizers" if name == "python" else "rust-tokenizers"), "runtime mismatch")
            require(run["fixture_sha256"] == hashlib.sha256(fixture).hexdigest() and run["manifest_sha256"] == manifest, "unpinned assets")
            require(run["profile"] == profile and run["strings_per_trial"] == 89, "wrong workload")
            require(run["source_sha256"] == groups[name][0]["source_sha256"], "changing sources")
            require([row["scope"] for row in run["measurements"]] == ["encode", "decode", "decode_skip_special"], "wrong scope")
    rng = {name: random.Random(20261110) for name in ("previous", "python")}
    rows = []
    scopes = ("encode", "decode", "decode_skip_special") + (("checked_load",) if include_load else ())
    for index, scope in enumerate(scopes):
        if scope == "checked_load":
            trials = {name: [r["load_trials_ms"] for r in group] for name, group in groups.items()}
        else:
            require(len({run["measurements"][index]["output_sha256"] for group in groups.values() for run in group}) == 1, "output mismatch")
            trials = {name: [r["measurements"][index]["trials_ms"] for r in group] for name, group in groups.items()}
        require(all(len(r) == 30 for group in trials.values() for r in group), "wrong trial count")
        summaries = {name: stats([v for r in group for v in r]) | {"process_medians_ms": [statistics.median(r) for r in group]}
                     for name, group in trials.items()}
        comparisons = {name: {"median_ratio": summaries["candidate"]["median_ms"]/summaries[name]["median_ms"],
                              "p95_ratio": summaries["candidate"]["p95_ms"]/summaries[name]["p95_ms"],
                              "paired_process_percentile_95_interval": interval(trials["candidate"], trials[name], rng[name])}
                       for name in ("previous", "python")}
        rows.append({"scope": scope, "timings": summaries, "candidate_over": comparisons})
    return {"schema_version": 1, "scope": "Controlled follow-up, not final performance acceptance; retains every declared process.",
            "processes_per_runtime": count, "cpu_affinity": environment["cpu_affinity"],
            **({"control_policy": "counterbalanced_fresh_processes; no CPU affinity on macOS"} if apple_control else {}),
            "native_binaries_identical": environment["binary_sha256"]["candidate"]["asset_bench"] == environment["binary_sha256"]["previous"]["asset_bench"],
            "uncertainty": "10000 paired whole-process bootstrap samples, seed 20261110 per comparator; descriptive 95% percentile intervals, not simultaneous guarantees.",
            "report_sha256": fingerprints, "measurements": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("bert", "gpt2", "qwen3", "tinystories"), default="tinystories")
    args = parser.parse_args()
    result = summarize(args.directory, args.profile)
    result["summarizer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False); file.write("\n")


if __name__ == "__main__":
    main()
