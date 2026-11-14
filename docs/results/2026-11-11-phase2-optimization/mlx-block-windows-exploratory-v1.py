"""Post-hoc, window-stratified B4 p95 analysis; not an acceptance audit.

Run from the repository root. It retains every declared process from both
completed controls and resamples whole paired process triplets within windows.
"""
import json
import math
import random
import tarfile

ARCHIVES = [
    ("earlier", "docs/results/2026-11-11-phase2-optimization/raw/tiny-borrow-apple-gate-v1.tar.xz", "reports/phase2-tiny-borrow-mlx-control-v1"),
    ("quiet", "docs/results/2026-11-11-phase2-optimization/raw/tiny-borrow-apple-quiet-gate-v1.tar.xz", "reports/phase2-tiny-borrow-mlx-quiet-control-v1"),
]


def q95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def load():
    windows = []
    for label, path, prefix in ARCHIVES:
        with tarfile.open(path) as archive:
            env = json.load(archive.extractfile(prefix + "/environment.json"))
            assert env["repeats"] == 9 and env["control_policy"] == "counterbalanced_fresh_processes"
            groups, sources = {}, {}
            for name in ("previous", "candidate", "python"):
                suffix = "python" if name == "python" else "rust"
                groups[name], sources[name] = [], []
                for i in range(1, 10):
                    report = json.load(archive.extractfile(f"{prefix}/{name}/codec-{suffix}-{i}.json"))
                    assert report["warmups"] == 5 and report["trials"] == 30
                    sources[name].append(report["source_sha256"])
                    trials = next(row["trials_ms"] for row in report["measurements"] if row["batch"] == 4 and row["scope"] == "resident_decoder_block")
                    assert len(trials) == 30 and all(math.isfinite(v) and v > 0 for v in trials)
                    groups[name].append(trials)
            windows.append((label, env, groups, sources))
    for key in ("binary_sha256", "native_libraries_sha256", "harness_sha256", "mlx-venv", "os", "rustc", "workloads", "cpu_affinity"):
        assert windows[0][1][key] == windows[1][1][key], key
    assert windows[0][3] == windows[1][3], "source inventories differ"
    return windows


def main():
    windows = load()
    for compare in ("previous", "python"):
        values = {name: [x for _, _, groups, _ in windows for trials in groups[name] for x in trials] for name in ("candidate", compare)}
        ratio = q95(values["candidate"]) / q95(values[compare])
        rng, samples = random.Random(20261114), []
        for _ in range(10000):
            selected = [rng.choices(range(9), k=9) for _ in windows]
            a = [v for (_, _, groups, _), indices in zip(windows, selected) for i in indices for v in groups["candidate"][i]]
            b = [v for (_, _, groups, _), indices in zip(windows, selected) for i in indices for v in groups[compare][i]]
            samples.append(q95(a) / q95(b))
        samples.sort()
        print(compare, "combined_p95_ratio", round(ratio, 6), "stratified_paired_process_percentile95_interval", round(samples[249], 6), round(samples[9749], 6), "n_per_runtime", 18)


if __name__ == "__main__":
    main()
