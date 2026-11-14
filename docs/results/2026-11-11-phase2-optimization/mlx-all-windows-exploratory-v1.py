"""Post-hoc two-window timing sensitivity, never an acceptance audit.

Run from the repository root with the pinned reference venv. Every raw control
report is hash-checked against its original audit, and all processes are kept.
"""
import hashlib
import json
import math
import random
import tarfile

import numpy as np

ARCHIVES = [
    ("docs/results/2026-11-11-phase2-optimization/raw/tiny-borrow-apple-gate-v1.tar.xz",
     "reports/phase2-tiny-borrow-mlx-control-v1", "reports/controlled-audit.json"),
    ("docs/results/2026-11-11-phase2-optimization/raw/tiny-borrow-apple-quiet-gate-v1.tar.xz",
     "reports/phase2-tiny-borrow-mlx-quiet-control-v1", "reports/quiet-controlled-audit.json"),
]
PROFILES = ("codec", "bert", "gpt2", "qwen3", "tinystories")
RUNTIMES = ("previous", "candidate", "python")


def q95(values):
    index = math.ceil(0.95 * len(values)) - 1
    return np.partition(values, index)[index]


def load():
    groups, sources, environments, audits = {}, {}, [], []
    for window, (path, prefix, audit_path) in enumerate(ARCHIVES):
        with tarfile.open(path) as archive:
            env = json.load(archive.extractfile(prefix + "/environment.json"))
            audit = json.load(archive.extractfile(audit_path))
            assert env["repeats"] == 9 and env["control_policy"] == "counterbalanced_fresh_processes"
            assert len(audit["cells"]) == 78 and audit["quality_passed"]
            environments.append(env)
            audits.append(audit)
            for profile in PROFILES:
                for runtime in RUNTIMES:
                    suffix = "python" if runtime == "python" else "rust"
                    stem = "codec" if profile == "codec" else f"tokenizer-{profile}"
                    for i in range(1, 10):
                        raw = archive.extractfile(f"{prefix}/{runtime}/{stem}-{suffix}-{i}.json").read()
                        assert hashlib.sha256(raw).hexdigest() == audit["controls"][profile]["report_sha256"][runtime][i - 1]
                        report = json.loads(raw)
                        assert report["trials"] == 30 and report["warmups"] == 5
                        key = (profile, runtime, i)
                        if window == 0:
                            sources[key] = report["source_sha256"]
                        else:
                            assert sources[key] == report["source_sha256"], key

                        def append(scope, values):
                            assert len(values) == 30 and all(math.isfinite(x) and x > 0 for x in values)
                            groups.setdefault((profile, scope, runtime), []).append(values)

                        append("checked_load", report["load_trials_ms"])
                        for measurement in report["measurements"]:
                            scope = (f"{measurement['batch']}/{measurement['scope']}" if profile == "codec"
                                     else measurement["scope"])
                            append(scope, measurement["trials_ms"])
    for key in ("binary_sha256", "native_libraries_sha256", "harness_sha256", "mlx-venv", "tokenizers-venv",
                "os", "rustc", "workloads", "cpu_affinity", "control_policy"):
        assert environments[0][key] == environments[1][key], key
    assert len(groups) == 29 * 3 and all(len(values) == 18 for values in groups.values())
    return groups, audits


def verify_original_points(groups, audits):
    for window, audit in enumerate(audits):
        checked = 0
        for row in audit["cells"]:
            if "controlled_followup" not in row:
                continue
            profile, scope = row["cell"].split("/", 1)
            candidate = np.asarray(groups[(profile, scope, "candidate")][window * 9:(window + 1) * 9]).reshape(-1)
            other = np.asarray(groups[(profile, scope, row["comparator"])][window * 9:(window + 1) * 9]).reshape(-1)
            control = row["controlled_followup"]
            assert abs(np.median(candidate) / np.median(other) - control["median_ratio"]) < 1e-12
            assert abs(q95(candidate) / q95(other) - control["p95_ratio"]) < 1e-12
            checked += 1
        memory = [row for row in audit["cells"] if "peak_observed_ratio" in row]
        assert checked == 58 and len(memory) == 20
        print("original window", window + 1, "point ratios verified", checked,
              "memory cells passed", sum(row["passed"] for row in memory), "/", len(memory))


def main():
    groups, audits = load()
    verify_original_points(groups, audits)
    rng = random.Random(20261114)
    indices = np.asarray([[*rng.choices(range(9), k=9),
                           *(i + 9 for i in rng.choices(range(9), k=9))]
                          for _ in range(10000)], dtype=np.intp)
    results = []
    for profile, scope in sorted({(p, s) for p, s, _ in groups}):
        candidate = np.asarray(groups[(profile, scope, "candidate")])
        for runtime in ("previous", "python"):
            other = np.asarray(groups[(profile, scope, runtime)])
            median_ratios, p95_ratios = [], []
            for offset in range(0, 10000, 1000):
                a = candidate[indices[offset:offset + 1000]].reshape(-1, 540)
                b = other[indices[offset:offset + 1000]].reshape(-1, 540)
                a = np.partition(a, (269, 270, 512), axis=1)
                b = np.partition(b, (269, 270, 512), axis=1)
                median_ratios.extend(((a[:, 269] + a[:, 270]) / (b[:, 269] + b[:, 270])).tolist())
                p95_ratios.extend((a[:, 512] / b[:, 512]).tolist())
            median_ratios.sort()
            p95_ratios.sort()
            row = {"cell": f"{profile}/{scope}", "comparator": runtime,
                   "median_ratio": float(np.median(candidate) / np.median(other)),
                   "p95_ratio": float(q95(candidate.reshape(-1)) / q95(other.reshape(-1))),
                   "median_upper": median_ratios[9749], "p95_upper": p95_ratios[9749]}
            row["within_bands"] = (row["median_ratio"] <= 1.05 and row["median_upper"] <= 1.05
                                   and row["p95_ratio"] <= 1.10 and row["p95_upper"] <= 1.10)
            results.append(row)
    assert len(results) == 58
    print("exploratory timing cells", len(results), "within bands", sum(row["within_bands"] for row in results))
    for row in results:
        if not row["within_bands"] or row["cell"] == "codec/4/resident_decoder_block":
            print(json.dumps(row, sort_keys=True))
    for field in ("median_upper", "p95_upper"):
        print("largest", field, json.dumps(max(results, key=lambda row: row[field]), sort_keys=True))


if __name__ == "__main__":
    main()
