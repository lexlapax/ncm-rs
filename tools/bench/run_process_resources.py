"""Separate whole-process CPU/RSS diagnostic; never acceptance wall-time evidence.

Run a JSON list of {name, argv, cwd} jobs in rotating fresh-process order.
Each argv must contain {report}, replaced with a unique benchmark output path.
CPU times include startup, loading, all trials, serialization and teardown. They
are not per-request times or allocation counts. wait4 RSS may include inherited
pre-exec high-water state; use benchmark-reported RSS for acceptance instead.
No profiler is injected.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def digest_file(path):
    # Do not inflate this launcher's resident heap by reading entire executables.
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def measure(argv, cwd, stdout, stderr, env):
    """wait4 accounts for this child, not unrelated or previously reaped children."""
    start = time.perf_counter()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        child = subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=err, env=env)
        try:
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        except BaseException:
            child.kill()
            child.wait()
            raise
    return {
        "exit_code": child.returncode,
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "user_cpu_ms": usage.ru_utime * 1000,
        "system_cpu_ms": usage.ru_stime * 1000,
        "wait4_peak_rss_bytes_including_pre_exec": usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def run(jobs, output, repeats=3, cpu=None):
    if platform.system() not in ("Linux", "Darwin"):
        raise ValueError("wait4/RSS units qualified only for Linux and Darwin")
    if not isinstance(repeats, int) or repeats < 3 or not jobs:
        raise ValueError("at least three repeats and one job required")
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names) or any(
        not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name)
        for name in names
    ):
        raise ValueError("unique safe job names required")
    if cpu is not None and (platform.system() != "Linux" or cpu not in os.sched_getaffinity(0)):
        raise ValueError("affinity requires an allowed Linux CPU")
    if any(os.environ.get(k) for k in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")):
        raise ValueError("use a separate instrumented profiling run")
    for job in jobs:
        if not isinstance(job["argv"], list) or not job["argv"] or not all(isinstance(a, str) for a in job["argv"]):
            raise ValueError("argv must be a nonempty list of strings")
        if sum(arg.count("{report}") for arg in job["argv"]) != 1:
            raise ValueError("each command needs exactly one {report} placeholder")
        if not Path(job["argv"][0]).is_file() or not Path(job["cwd"]).is_dir():
            raise ValueError("explicit existing executable and cwd required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "status": "incomplete",
        "scope": "Whole process, including startup/load/all benchmark trials/reporting/exit; not per-request CPU or allocation counts.",
        "memory_limitation": "wait4 high-water RSS may include inherited pre-exec state. Use benchmark-reported RSS/HWM for acceptance; do not equate these counters.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "cpu_affinity": cpu,
        "repeats": repeats,
        "source_sha256": digest_file(__file__),
        "jobs": jobs,
        "binary_sha256": {j["name"]: digest_file(j["argv"][0]) for j in jobs},
        "runs": [],
    }
    def save():
        (output / "resources.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    for repeat in range(1, repeats + 1):
        offset = (repeat - 1) % len(jobs)
        for job in jobs[offset:] + jobs[:offset]:
            stem = f"{job['name']}-{repeat}"
            report = output / f"{stem}.json"
            argv = [a.replace("{report}", str(report)) for a in job["argv"]]
            if cpu is not None:
                argv = ["taskset", "-c", str(cpu)] + argv
            row = {"name": job["name"], "repeat": repeat, "argv": argv,
                   "started_utc": datetime.now(timezone.utc).isoformat()}
            row.update(measure(argv, job["cwd"], output / f"{stem}.stdout", output / f"{stem}.stderr", env))
            result["runs"].append(row)
            save()
            if row["exit_code"] != 0:
                raise RuntimeError(f"{stem} failed; diagnostic retained")
            # Parse the child JSON, but do not claim workload/quality qualification.
            json.loads(report.read_bytes())
            row["report_sha256"] = digest_file(report)
            save()
    result["status"] = "complete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    run(json.loads(args.commands.read_bytes()), args.output, args.repeats, args.cpu)


if __name__ == "__main__":
    main()
