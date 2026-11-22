"""Run alternating fresh-process before/after benchmarks for Phase 5 Step 3."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

WORKLOADS = [
    ("sudoku-easy", "sudoku-easy-qualified", 1, "artifacts/canonical/sudoku-data-phase3/easy-128-inputs.json"),
    ("tinystories", "tinystories-phase4-v1", 1, None),
    ("xsum", "xsum-inference-rope-v2-final", 1, "artifacts/canonical/text-phase4-v1/xsum.json"),
    ("wmt14", "wmt14-inference-v1-final", 1, "reports/phase4-wmt14-tokenizer-reference-v1.json"),
    ("lm1b-small", "lm1b-small-inference-v1", 1, "artifacts/canonical/lm1b-reference-rope-v1.safetensors"),
    ("lm1b-medium", "lm1b-medium-inference-v1", 1, "artifacts/canonical/lm1b-reference-rope-v1.safetensors"),
    ("lm1b-large", "lm1b-large-inference-v1", 1, "artifacts/canonical/lm1b-reference-rope-v1.safetensors"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_metrics(rec: dict, platform: str) -> dict:
    load = rec["model_load"]
    load_art = load["after_artifact_load"]
    load_up = load["after_weight_upload"]
    uninst = rec["uninstrumented"]
    uninst_mem = uninst["after_memory"]
    out = {
        "model_load_total_ms": load["total_ms"],
        "artifact_load_ms": load["artifact_load_ms"],
        "weight_upload_ms": load["weight_upload_ms"],
        "rss_after_artifact_mib": load_art["host"]["VmRSS"] / (1024 * 1024),
        "rss_after_upload_mib": load_up["host"]["VmRSS"] / (1024 * 1024),
        "hwm_mib": uninst_mem["host"]["VmHWM"] / (1024 * 1024),
        "warm_median_ms": uninst["median_ms"],
        "warm_p95_ms": uninst["p95_ms"],
        "warm_trials_ms": uninst["trials_ms"],
    }
    if platform == "cuda":
        out["device_peak_mib"] = uninst_mem["cuda_default_pool_bytes"]["used_peak"] / (1024 * 1024)
        out["device_reserved_peak_mib"] = (
            uninst_mem["cuda_default_pool_bytes"]["reserved_peak"] / (1024 * 1024)
        )
    else:
        out["device_active_after_load_mib"] = load_up["mlx_memory_bytes"]["active"] / (1024 * 1024)
        out["device_peak_mib"] = uninst_mem["mlx_memory_bytes"]["peak"] / (1024 * 1024)
    return out


def aggregate(runs: list[dict], platform: str) -> dict:
    def med(key: str) -> float:
        return round(statistics.median(r[key] for r in runs), 2)

    all_warm = sorted(x for r in runs for x in r["warm_trials_ms"])
    p95_idx = max(0, int(0.95 * len(all_warm) + 0.999999) - 1)
    agg = {
        "processes": len(runs),
        "model_load_total_ms_median": med("model_load_total_ms"),
        "model_load_total_ms_runs": [round(r["model_load_total_ms"], 2) for r in runs],
        "artifact_load_ms_median": med("artifact_load_ms"),
        "weight_upload_ms_median": med("weight_upload_ms"),
        "rss_after_artifact_mib_median": med("rss_after_artifact_mib"),
        "rss_after_upload_mib_median": med("rss_after_upload_mib"),
        "hwm_mib_median": med("hwm_mib"),
        "hwm_mib_runs": [round(r["hwm_mib"], 2) for r in runs],
        "warm_trials_total": len(all_warm),
        "warm_median_ms": round(statistics.median(all_warm), 2),
        "warm_p95_ms": round(all_warm[p95_idx], 2),
        "device_peak_mib_median": med("device_peak_mib"),
    }
    if platform == "cuda":
        agg["device_reserved_peak_mib_median"] = med("device_reserved_peak_mib")
    else:
        agg["device_active_after_load_mib_median"] = med("device_active_after_load_mib")
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cuda", "mlx"), required=True)
    parser.add_argument("--before-binary", type=Path, required=True)
    parser.add_argument("--after-binary", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.processes < 3:
        raise ValueError("processes must be >= 3")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output directory already exists")
    args.output.mkdir(parents=True, exist_ok=False)

    root = Path.cwd()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    rows = []
    for profile, art_dir, batch, extra in WORKLOADS:
        art_path = root / "artifacts/canonical" / art_dir
        runs = {"before": [], "after": []}
        for rep in range(args.processes):
            for label, binary in (("before", args.before_binary), ("after", args.after_binary)):
                argv = [str(binary), profile, str(art_path), str(batch)]
                if extra is not None:
                    argv.append(str(root / extra))
                proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600, check=False)
                if proc.returncode != 0:
                    raise RuntimeError(f"{profile} {label} rep={rep} failed ({proc.returncode}): {proc.stderr}")
                raw_path = args.output / f"{profile}-{label}-rep{rep}.json"
                raw_path.write_text(proc.stdout, encoding="utf-8")
                runs[label].append(extract_metrics(json.loads(proc.stdout), args.platform))
        b = aggregate(runs["before"], args.platform)
        a = aggregate(runs["after"], args.platform)
        load_speedup = round(b["model_load_total_ms_median"] / a["model_load_total_ms_median"], 3)
        hwm_ratio = round(a["hwm_mib_median"] / b["hwm_mib_median"], 3)
        warm_ratio = round(a["warm_median_ms"] / b["warm_median_ms"], 3)
        rows.append(
            {
                "profile": profile,
                "batch": batch,
                "before": b,
                "after": a,
                "load_speedup_x": load_speedup,
                "hwm_ratio_after_over_before": hwm_ratio,
                "hwm_saved_mib": round(b["hwm_mib_median"] - a["hwm_mib_median"], 2),
                "warm_ratio_after_over_before": warm_ratio,
            }
        )
        print(
            f"{profile:12s}  load: {b['model_load_total_ms_median']:7.1f} -> {a['model_load_total_ms_median']:7.1f} ms "
            f"({load_speedup:.2f}x)  HWM: {b['hwm_mib_median']:6.0f} -> {a['hwm_mib_median']:6.0f} MiB "
            f"({hwm_ratio:.2f}x, -{b['hwm_mib_median'] - a['hwm_mib_median']:.0f} MiB)  "
            f"warm: {b['warm_median_ms']:6.2f} -> {a['warm_median_ms']:6.2f} ms ({warm_ratio:.3f}x)",
            flush=True,
        )

    summary = {
        "platform": args.platform,
        "processes_per_side": args.processes,
        "alternation_order": "before, after (repeated)",
        "before_binary_sha256": sha256(args.before_binary),
        "after_binary_sha256": sha256(args.after_binary),
        "rows": rows,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY_SHA256:", sha256(summary_path))


if __name__ == "__main__":
    main()
