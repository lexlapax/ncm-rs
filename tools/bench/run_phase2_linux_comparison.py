"""Interleave previous Rust, candidate Rust and Python in fresh processes on Linux.

Uses unchanged Phase 2 workload/scopes. Binaries must be built before invocation.
One Python sample is shared between the two summaries, not counted twice as evidence.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from sample_linux_cpu import cpu_list

ROOT = Path(__file__).resolve().parents[2]


def capture(command, cwd=ROOT):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def order(repeat):
    names = ("previous", "python", "candidate")
    offset = (repeat - 1) % len(names)
    return names[offset:] + names[:offset]


def validate_observer(cpu, observer_cpu, sys_cpu=Path('/sys/devices/system/cpu')):
    """Check placement before creating output or narrowing the launcher's affinity."""
    if observer_cpu is None:
        return
    allowed = os.sched_getaffinity(0)
    if cpu is None or cpu not in allowed or observer_cpu not in allowed:
        raise ValueError('observer requires an explicit workload CPU and permitted CPUs')
    siblings = cpu_list((sys_cpu/f'cpu{cpu}/topology/thread_siblings_list').read_text())
    if cpu not in siblings:
        raise ValueError('inconsistent SMT topology')
    if observer_cpu in siblings:
        raise ValueError('observer must not run on workload CPU or its SMT sibling')


@contextmanager
def observer_affinity(cpu):
    """Monitors inherit this placement; taskset still places each workload separately."""
    if cpu is None:
        yield
        return
    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cpu})
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_checkout", type=Path)
    parser.add_argument("previous_binaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", choices=("codec", "bert", "gpt2", "qwen3", "tinystories"),
                        default=["codec", "bert", "gpt2", "qwen3", "tinystories"])
    parser.add_argument("--cpu", type=int, help="same process-local CPU affinity for all three runtimes")
    parser.add_argument("--observer-cpu", type=int,
                        help="pin launcher and GPU observers away from the workload's SMT siblings; requires --cpu")
    parser.add_argument("--identical-tokenizer-binary", action="store_true",
                        help="control executable-path effects only after verifying byte-identical native binaries")
    args = parser.parse_args()
    previous = args.previous_checkout.resolve()
    old_bins = args.previous_binaries.resolve()
    output = args.output.resolve()
    if args.repeats < 3 or len(set(args.workloads)) != len(args.workloads):
        raise ValueError("at least three processes and unique workloads required")
    if args.cpu is not None and args.cpu not in os.sched_getaffinity(0):
        raise ValueError("CPU outside allowed affinity")
    if platform.system() != "Linux":
        raise ValueError("Linux only")
    validate_observer(args.cpu, args.observer_cpu)
    if any(os.environ.get(k) for k in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH", "LD_PRELOAD")):
        raise ValueError("profiling/injection environment is not acceptance timing")
    if capture(["git", "status", "--porcelain"], previous):
        raise ValueError("previous source checkout must be clean")
    bins = {"previous": old_bins, "candidate": ROOT / "target/release/examples"}
    fingerprints = {}
    for runtime, directory in bins.items():
        fingerprints[runtime] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                 for name in ("codec_bench", "asset_bench")}
    if args.identical_tokenizer_binary:
        if "codec" in args.workloads or fingerprints["previous"]["asset_bench"] != fingerprints["candidate"]["asset_bench"]:
            raise ValueError("identity control requires tokenizer-only work and byte-identical binaries")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("previous", "candidate", "python"):
        (output/name).mkdir()
    environment = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_heads": {"previous": capture(["git", "rev-parse", "HEAD"], previous),
                         "candidate": capture(["git", "rev-parse", "HEAD"])},
        "candidate_status": capture(["git", "status", "--short"]),
        "binary_sha256": fingerprints, "rustc": capture(["rustc", "--version"]),
        "kernel": platform.release(), "cpu": capture(["lscpu"]),
        "cuda_compiler": capture(["nvcc", "--version"]),
        "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,power.limit,utilization.gpu", "--format=csv"]),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repeats": args.repeats, "workloads": args.workloads, "cpu_affinity": args.cpu,
        "observer_cpu": args.observer_cpu,
        "observer_helper_sha256": hashlib.sha256(Path(__file__).with_name('sample_linux_cpu.py').read_bytes()).hexdigest(),
        "identical_tokenizer_binary_control": args.identical_tokenizer_binary,
        "commands": [], "paired_python": "Same Python report reused for previous/candidate comparison; not independent duplicates.",
        "limitations": ["Live desktop; telemetry retained. No clock/power/global tooling changes.",
                        "Warm filesystem cache; existing harness excludes Python imports and CUDA context startup.",
                        "Existing unchanged output retention, transfers, warmups and timing boundaries.",
                        "Three process orders rotate previous/python/candidate; individual trials are not independent processes."],
    }
    for venv in ("venv", "tokenizers-venv"):
        python = str(ROOT/f".reference/{venv}/bin/python")
        environment[venv] = {"python": capture([python, "--version"]),
                             "packages": capture([python, "-m", "pip", "freeze"])}
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    def save():
        (output/"environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    def run(command, cwd, runtime, report):
        if args.cpu is not None:
            command = ["taskset", "-c", str(args.cpu)] + command
        entry = {"argv": command, "cwd": str(cwd), "runtime": runtime,
                 "started_utc": datetime.now(timezone.utc).isoformat()}
        environment["commands"].append(entry)
        save()
        subprocess.run(command, cwd=cwd, env=env, check=True)
        entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
        entry["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        save()
    with observer_affinity(args.observer_cpu), (output/"gpu-telemetry.csv").open("x") as telemetry, (output/"gpu-process-utilization.txt").open("x") as utilization:
        environment["launcher_affinity"] = sorted(os.sched_getaffinity(0))
        save()
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv", "-l", "1"], stdout=telemetry)
        processes = subprocess.Popen(["nvidia-smi", "pmon", "-s", "u", "-d", "1"], stdout=utilization)
        try:
            for repeat in range(1, args.repeats + 1):
                for profile in args.workloads:
                    profile = None if profile == "codec" else profile
                    prefix = "codec" if profile is None else f"tokenizer-{profile}"
                    for runtime in order(repeat):
                        report = output/runtime/f"{prefix}-{'python' if runtime == 'python' else 'rust'}-{repeat}.json"
                        cwd = previous if runtime == "previous" else ROOT
                        if profile is None:
                            if runtime == "python":
                                command = [str(ROOT/".reference/venv/bin/python"), "tools/reference/codec_bench.py",
                                           str(ROOT/"artifacts/canonical/sudoku-easy-qualified"), str(ROOT/".reference/upstream-ncm")]
                            else:
                                command = [str(bins[runtime]/"codec_bench"), str(ROOT/"artifacts/canonical/sudoku-easy-qualified")]
                            command += ["tests/fixtures/codec-bench-v1.json", str(report)]
                        else:
                            if runtime == "python":
                                command = [str(ROOT/".reference/tokenizers-venv/bin/python"), "tools/reference/text_tokenizer_bench.py",
                                           str(ROOT/"artifacts/upstream/tokenizers-v2"), str(ROOT/".reference/upstream-ncm")]
                            else:
                                binary = bins["candidate" if args.identical_tokenizer_binary else runtime]/"asset_bench"
                                command = [str(binary), str(ROOT/"artifacts/canonical/tokenizers-qualified")]
                            command += ["tests/fixtures/tokenizers-v1.json", profile, str(report)]
                        run(command, cwd, runtime, report)
                        if runtime == "python":
                            for destination in ("previous", "candidate"):
                                shutil.copyfile(report, output/destination/report.name)
            if monitor.poll() is not None or processes.poll() is not None:
                raise RuntimeError("telemetry process exited early")
        finally:
            monitor.terminate(); processes.terminate(); monitor.wait(); processes.wait()
            environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()
    print("Three-way measurements complete:", output)


if __name__ == "__main__":
    main()
