"""Sequential alternating Apple measurements; no global tooling/power changes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[2]


def capture(*command):
    return subprocess.check_output(command,text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("state_root",type=Path,help="checkout holding native libraries, venvs, assets and built binaries")
    p.add_argument("output",type=Path);a=p.parse_args();a.state_root=a.state_root.resolve();a.output=a.output.resolve()
    if platform.system()!="Darwin" or platform.machine()!="arm64":raise ValueError("requires Apple Silicon")
    os.chdir(ROOT);a.output.mkdir(parents=True,exist_ok=False)
    state=a.state_root
    environment={"started_utc":datetime.now(timezone.utc).isoformat(),"source_git_head":capture("git","rev-parse","HEAD"),
        "source_git_status":capture("git","status","--short"),"rustc":capture("rustc","--version"),"cargo":capture("cargo","--version"),
        "os":capture("sw_vers"),"xcode":capture("xcodebuild","-version"),"cpu":capture("sysctl","-n","machdep.cpu.brand_string"),
        "memory_bytes":int(capture("sysctl","-n","hw.memsize")),"physical_cpus":int(capture("sysctl","-n","hw.physicalcpu")),
        "power_settings":capture("pmset","-g","custom"),"thermal_status_before":capture("pmset","-g","therm"),"commands":[],
        "scope_notes":["Workstation desktop activity remains; no exclusive hardware reservation, global tooling or power changes.",
            "CPU activity sampled every two seconds; GPU temperature/utilization not available without privileged tooling, not invented.",
            "Native binaries run with cleared environment and PATH=/usr/bin:/bin; Python is offline orchestration/reference only.",
            "Warm filesystem cache; context/import startup excluded; graph construction and synchronous evaluation included.",
            "Unified MLX active/peak/cache and process RSS/HWM are distinct metrics, not separate VRAM."],
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    paths=[state/"target/release/examples/mlx_codec_bench",state/"target/release/examples/asset_bench",state/".reference/mlx-native/lib/libmlx.dylib",state/".reference/mlx-native/lib/mlx.metallib"]
    environment["binary_sha256"]={str(path.relative_to(state)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for venv in ("mlx-venv","tokenizers-venv"):
        python=str(state/f".reference/{venv}/bin/python")
        environment[venv]={"python":capture(python,"--version"),"packages":json.loads(capture(python,"-c","import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"))}
    stop=threading.Event()
    def sampler():
        with (a.output/"cpu-activity.jsonl").open("x") as file:
            while not stop.is_set():
                entries=[]
                for line in capture("ps","-Ao","pid=,pcpu=,comm=").splitlines():
                    pid,cpu,name=line.split(None,2);entries.append({"pid":int(pid),"cpu_percent":float(cpu),"name":Path(name).name})
                row={"utc":datetime.now(timezone.utc).isoformat(),"load_average":os.getloadavg(),"top_cpu":sorted(entries,key=lambda r:r["cpu_percent"],reverse=True)[:12]}
                file.write(json.dumps(row)+"\n");file.flush();stop.wait(2)
    def save():
        (a.output/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    def run(command,native):
        environment["commands"].append({"argv":command,"cleared_environment":native});save()
        env={"PATH":"/usr/bin:/bin"} if native else {**os.environ,"TOKENIZERS_PARALLELISM":"false","HF_HUB_OFFLINE":"1"}
        subprocess.run(command,check=True,env=env)
    def monitored_sampler():
        try:sampler()
        except Exception as exc:environment["telemetry_error"]=str(exc)
    thread=threading.Thread(target=monitored_sampler);thread.start()
    try:
        for repeat in range(1,4):
            order=["python","rust"] if repeat%2 else ["rust","python"]
            for runtime in order:
                output=str(a.output/f"codec-{runtime}-{repeat}.json")
                if runtime=="rust":command=[str(state/"target/release/examples/mlx_codec_bench"),str(state/"artifacts/canonical/sudoku-easy-qualified"),"tests/fixtures/codec-bench-v1.json",output]
                else:command=[str(state/".reference/mlx-venv/bin/python"),"tools/reference/mlx_codec_bench.py",str(state/"artifacts/canonical/sudoku-easy-qualified"),"tests/fixtures/codec-bench-v1.json",str(state/"reports/phase2-codec/python-mlx-v1"),output]
                run(command,runtime=="rust")
            for profile in ("bert","gpt2","qwen3","tinystories"):
                for runtime in order:
                    output=str(a.output/f"tokenizer-{profile}-{runtime}-{repeat}.json")
                    if runtime=="rust":command=[str(state/"target/release/examples/asset_bench"),str(state/"artifacts/canonical/tokenizers-qualified")]
                    else:command=[str(state/".reference/tokenizers-venv/bin/python"),"tools/reference/text_tokenizer_bench.py",str(state/"artifacts/upstream/tokenizers-v2"),str(state/".reference/upstream-ncm")]
                    run(command+["tests/fixtures/tokenizers-v1.json",profile,output],runtime=="rust")
    finally:
        stop.set();thread.join();environment["finished_utc"]=datetime.now(timezone.utc).isoformat();environment["thermal_status_after"]=capture("pmset","-g","therm");save()
    if "telemetry_error" in environment:raise RuntimeError(environment["telemetry_error"])
    print("Measurements complete:",a.output)


if __name__=="__main__":main()
