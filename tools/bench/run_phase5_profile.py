"""Run Phase 5 Step 2 stage-breakdown profiling across all released profiles on CUDA or MLX."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

WORKLOADS = [
    ("sudoku-easy", "sudoku-easy-qualified", [1, 128], "artifacts/canonical/sudoku-data-phase3/easy-128-inputs.json"),
    ("tinystories", "tinystories-phase4-v1", [1, 128], None),
    ("xsum", "xsum-inference-rope-v2-final", [1, 32], "artifacts/canonical/text-phase4-v1/xsum.json"),
    ("wmt14", "wmt14-inference-v1-final", [1, 32], "reports/phase4-wmt14-tokenizer-reference-v1.json"),
    ("lm1b-small", "lm1b-small-inference-v1", [1, 16], "artifacts/canonical/lm1b-reference-rope-v1.safetensors"),
    ("lm1b-medium", "lm1b-medium-inference-v1", [1, 16], "artifacts/canonical/lm1b-reference-rope-v1.safetensors"),
    ("lm1b-large", "lm1b-large-inference-v1", [1, 8], "artifacts/canonical/lm1b-reference-rope-v1.safetensors"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mem_summary(record: dict, platform: str) -> dict:
    load_art = record["model_load"]["after_artifact_load"]
    load_up = record["model_load"]["after_weight_upload"]
    uninst = record["uninstrumented"]["after_memory"]
    hwm_mib = uninst["host"]["VmHWM"] / (1024 * 1024)
    rss_load_mib = load_up["host"]["VmRSS"] / (1024 * 1024)
    rss_run_mib = uninst["host"]["VmRSS"] / (1024 * 1024)
    out = {
        "rss_after_artifact_mib": round(load_art["host"]["VmRSS"] / (1024 * 1024), 2),
        "rss_after_upload_mib": round(rss_load_mib, 2),
        "rss_after_run_mib": round(rss_run_mib, 2),
        "hwm_mib": round(hwm_mib, 2),
    }
    if platform == "cuda":
        pool_load = load_up["cuda_default_pool_bytes"]
        pool_run = uninst["cuda_default_pool_bytes"]
        out["cuda_pool_used_peak_load_mib"] = round(pool_load["used_peak"] / (1024 * 1024), 2)
        out["cuda_pool_used_peak_run_mib"] = round(pool_run["used_peak"] / (1024 * 1024), 2)
        out["cuda_pool_reserved_peak_run_mib"] = round(pool_run["reserved_peak"] / (1024 * 1024), 2)
    else:
        mlx_load = load_up["mlx_memory_bytes"]
        mlx_run = uninst["mlx_memory_bytes"]
        out["mlx_active_after_load_mib"] = round(mlx_load["active"] / (1024 * 1024), 2)
        out["mlx_peak_run_mib"] = round(mlx_run["peak"] / (1024 * 1024), 2)
        out["mlx_cache_run_mib"] = round(mlx_run["cache"] / (1024 * 1024), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cuda", "mlx"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output directory already exists")
    args.output.mkdir(parents=True, exist_ok=False)

    root = Path.cwd()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    rows = []
    for profile, art_dir, batches, extra in WORKLOADS:
        art_path = root / "artifacts/canonical" / art_dir
        for batch in batches:
            argv = [str(args.binary), profile, str(art_path), str(batch)]
            if extra is not None:
                argv.append(str(root / extra))
            proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"{profile} b={batch} failed ({proc.returncode}): {proc.stderr}")
            out_file = args.output / f"{profile}-b{batch}.json"
            out_file.write_text(proc.stdout, encoding="utf-8")
            rec = json.loads(proc.stdout)
            st = rec["stages"]
            tot = st["instrumented_total_ms"]
            row = {
                "profile": profile,
                "batch": batch,
                "model_load_total_ms": round(rec["model_load"]["total_ms"], 2),
                "artifact_load_ms": round(rec["model_load"]["artifact_load_ms"], 2),
                "weight_upload_ms": round(rec["model_load"]["weight_upload_ms"], 2),
                "uninstrumented_median_ms": round(rec["uninstrumented"]["median_ms"], 2),
                "uninstrumented_p95_ms": round(rec["uninstrumented"]["p95_ms"], 2),
                "instrumented_total_ms": round(tot, 2),
                "prefix_prepare_ms": round(st["prefix_prepare_ms"], 2),
                "generator_levels_sum_ms": round(st["generator_levels_sum_ms"], 2),
                "per_level_sampling_ms": round(st["per_level_sampling_ms"], 2),
                "final_penalties_ms": round(st["final_penalties_ms"], 2),
                "sampling_and_penalties_ms": round(st["sampling_and_penalties_ms"], 2),
                "codec_decode_ms": round(st["codec_decode_ms"], 2),
                "host_readback_ms": round(st["host_readback_ms"], 2),
                "memory": mem_summary(rec, args.platform),
                "per_level_generator_ms": st["per_level_generator_ms"],
            }
            rows.append(row)
            print(
                f"{profile:12s} b={batch:3d}  load={row['model_load_total_ms']:7.1f}ms "
                f"(art={row['artifact_load_ms']:6.1f} up={row['weight_upload_ms']:6.1f})  "
                f"warm={row['uninstrumented_median_ms']:7.2f}ms  "
                f"[pref={row['prefix_prepare_ms']:5.2f} gen={row['generator_levels_sum_ms']:7.2f} "
                f"samp+pen={row['sampling_and_penalties_ms']:6.2f} codec={row['codec_decode_ms']:6.2f} "
                f"read={row['host_readback_ms']:5.2f}]  hwm={row['memory']['hwm_mib']:.0f}MiB",
                flush=True,
            )

    summary = {
        "platform": args.platform,
        "binary_sha256": sha256(args.binary),
        "rows": rows,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY_SHA256:", sha256(summary_path))


if __name__ == "__main__":
    main()
