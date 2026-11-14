"""One fixed three-pair full-codec diagnostic; not an acceptance gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time


root = Path.home() / "projects/lexlapax/ncm-rs"
experiment = root / ".reference/phase2-mlx-gelu-codec-v1"
output = root / "reports/phase2-mlx-gelu-codec-diagnostic-v1"
output.mkdir(exist_ok=False)
binaries = {
    "current": root / "target/release/examples/mlx_codec_bench",
    "compiled": root / ".reference/phase2-mlx-gelu-codec-target-v1/release/examples/mlx_codec_bench",
}
metadata = {
    "plan": "Three alternating fresh-process pairs; all codec scopes; not acceptance.",
    "binaries": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()},
    "commands": [],
    "thermal_before": subprocess.check_output(["pmset", "-g", "therm"], text=True),
}
stop = threading.Event()


def sample():
    try:
        with (output / "activity.jsonl").open("x") as file:
            while not stop.is_set():
                entries = []
                for line in subprocess.check_output(["ps", "-Ao", "pcpu=,comm="], text=True).splitlines():
                    cpu, name = line.split(None, 1)
                    entries.append({"cpu_percent": float(cpu), "name": Path(name).name})
                file.write(json.dumps({"time": time.time(), "top_cpu": sorted(entries, key=lambda x: x["cpu_percent"], reverse=True)[:12]}) + "\n")
                file.flush()
                stop.wait(2)
    except Exception as exc:
        metadata["activity_error"] = str(exc)


thread = threading.Thread(target=sample)
thread.start()
try:
    for repeat in range(1, 4):
        for name in (("current", "compiled") if repeat % 2 else ("compiled", "current")):
            command = [str(binaries[name]), str(root / "artifacts/canonical/sudoku-easy-qualified"),
                       "tests/fixtures/codec-bench-v1.json", str(output / f"{name}-{repeat}.json")]
            cwd = root if name == "current" else experiment
            entry = {"argv": command, "cwd": str(cwd), "start": time.time()}
            metadata["commands"].append(entry)
            (output / "plan.json").write_text(json.dumps(metadata, indent=2) + "\n")
            subprocess.run(command, cwd=cwd, env={"PATH": "/usr/bin:/bin"}, timeout=120, check=True)
            entry["end"] = time.time()
finally:
    stop.set()
    thread.join()
    metadata["thermal_after"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
    (output / "plan.json").write_text(json.dumps(metadata, indent=2) + "\n")
if "activity_error" in metadata:
    raise RuntimeError(metadata["activity_error"])
