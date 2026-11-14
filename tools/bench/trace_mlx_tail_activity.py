"""Run one bounded, system-wide Metal trace around the marked diagnostic.

The full trace and recorder output stay in the caller-selected local directory;
they may contain unrelated process metadata. This is not an acceptance runner.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkout, binary, bundle, output = (p.resolve() for p in (args.checkout, args.binary, args.bundle, args.output))
    if os.uname().sysname != "Darwin" or not binary.is_file():
        raise RuntimeError("Apple diagnostic executable required")
    output.mkdir(parents=True, exist_ok=False)
    trace, log, report = output / "system.trace", output / "recorder.log", output / "report.json"
    command = ["/usr/bin/xcrun", "xctrace", "record", "--template", "Metal System Trace",
               "--instrument", "Points of Interest", "--instrument", "Thread Activity",
               "--all-processes", "--no-prompt", "--time-limit", "35s", "--output", str(trace)]
    target = [str(binary), str(bundle), "tests/fixtures/codec-bench-v1.json", str(report)]
    metadata = {"trace_command": command, "target_command": target, "target_cwd": str(checkout),
                "trace_scope": "system-wide; raw trace remains private", "status": "starting"}

    def save():
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    recorder = None
    try:
        with log.open("xb", buffering=0) as stream:
            recorder = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if recorder.poll() is not None:
                    raise RuntimeError(f"recorder exited before readiness: {recorder.returncode}")
                if b"Starting recording" in log.read_bytes():
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("recorder readiness not observed")
            metadata["status"] = "target_running"
            save()
            completed = subprocess.run(target, cwd=checkout, env={"PATH": "/usr/bin:/bin"},
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=stream, timeout=25, check=False)
            metadata["target_returncode"] = completed.returncode
            # The recording itself has a fixed 35-second limit. Saving a
            # system-wide trace can take longer; do not destroy a live save
            # merely because an arbitrary post-target wait expired.
            metadata["status"] = "saving"
            save()
            metadata["recorder_returncode"] = recorder.wait()
            metadata["status"] = "complete" if completed.returncode == 0 and recorder.returncode == 0 else "failed"
            save()
            if metadata["status"] != "complete":
                raise RuntimeError("target or recorder failed; preserve outputs")
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
            try:
                recorder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()
        if metadata["status"] not in ("complete", "failed"):
            metadata["status"] = "failed"
            save()


if __name__ == "__main__":
    main()
