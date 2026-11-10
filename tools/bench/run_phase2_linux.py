"""Sequential alternating fresh-process Phase 2 measurements; no builds or downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def capture(*command):
    return subprocess.check_output(command,text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("output",type=Path); a=p.parse_args()
    os.chdir(ROOT)
    if platform.system()!="Linux": raise ValueError("Linux only")
    a.output.mkdir(parents=True,exist_ok=False)
    environment={"started_utc":datetime.now(timezone.utc).isoformat(),"git_head":capture("git","rev-parse","HEAD"),
        "git_status":capture("git","status","--short"),"rustc":capture("rustc","--version"),"cargo":capture("cargo","--version"),
        "kernel":platform.release(),"cpu":next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        "gpu":capture("nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.used,power.limit,utilization.gpu","--format=csv"),
        "cuda_compiler":capture("nvcc","--version"),"commands":[],
        "scope_notes":["Workstation GPU; existing resident applications are not stopped or reconfigured. See telemetry/pmon for activity.",
            "No clock/power controls changed. Warm filesystem cache, not disk-cold load. Python import and CUDA context startup excluded.",
            "Codec runtime allocations and device-global observations differ; do not interpret global usage as process peak.",
            "CPU tokenizer profiles use separate fresh processes; Python source loading and native canonical loading differ in representation."],
        "binaries_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["target/release/examples/codec_bench","target/release/examples/asset_bench"]}}
    governor=Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    environment["cpu_governor"]=governor.read_text().strip() if governor.exists() else None
    for venv in ["venv","tokenizers-venv"]:
        environment[venv]={"python":capture(f".reference/{venv}/bin/python","--version"),"packages":capture(f".reference/{venv}/bin/python","-m","pip","freeze")}
    def save():
        (a.output/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    def run(command):
        environment["commands"].append(command);save()
        subprocess.run(command,check=True)
    env=os.environ.copy();env["TOKENIZERS_PARALLELISM"]="false";os.environ.update(env)
    with (a.output/"gpu-telemetry.csv").open("w") as telemetry,(a.output/"gpu-process-utilization.txt").open("w") as utilization:
        monitor=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem","--format=csv","-l","1"],stdout=telemetry)
        processes=subprocess.Popen(["nvidia-smi","pmon","-s","u","-d","1"],stdout=utilization)
        try:
            for repeat in range(1,4):
                for runtime in (["python","rust"] if repeat%2 else ["rust","python"]):
                    output=str(a.output/f"codec-{runtime}-{repeat}.json")
                    command=([".reference/venv/bin/python","tools/reference/codec_bench.py","artifacts/canonical/sudoku-easy-qualified",".reference/upstream-ncm"] if runtime=="python" else ["target/release/examples/codec_bench","artifacts/canonical/sudoku-easy-qualified"])
                    run(command+["tests/fixtures/codec-bench-v1.json",output])
                for profile in ["bert","gpt2","qwen3","tinystories"]:
                    for runtime in (["python","rust"] if repeat%2 else ["rust","python"]):
                        output=str(a.output/f"tokenizer-{profile}-{runtime}-{repeat}.json")
                        command=([".reference/tokenizers-venv/bin/python","tools/reference/text_tokenizer_bench.py","artifacts/upstream/tokenizers-v2",".reference/upstream-ncm"] if runtime=="python" else ["target/release/examples/asset_bench","artifacts/canonical/tokenizers-qualified"])
                        run(command+["tests/fixtures/tokenizers-v1.json",profile,output])
        finally:
            monitor.terminate();processes.terminate();monitor.wait();processes.wait()
            environment["finished_utc"]=datetime.now(timezone.utc).isoformat();save()
    print("Measurements complete:",a.output)


if __name__=="__main__":main()
