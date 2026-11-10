"""Trusted canonical-artifact load benchmark; not a production/security validator."""
import argparse
import gc
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import torch
from safetensors.torch import load


def checked_load(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    payloads = {}
    for name, record in manifest["files"].items():
        data = (root / name).read_bytes()
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
        if name.endswith(".json"):
            json.loads(data)
        payloads[name] = data
    arrays = []
    for name, records in manifest["tensors"].items():
        data = payloads[name]
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + size])
        assert set(header) - {"__metadata__"} == {r["name"] for r in records}
        for record in records:
            entry = header[record["name"]]
            begin, end = entry["data_offsets"]
            raw = memoryview(data)[8 + size + begin:8 + size + end]
            assert entry["shape"] == record["shape"] and entry["dtype"] == record["dtype"]
            assert math.prod(entry["shape"]) * {"F32": 4, "I64": 8}[entry["dtype"]] == record["bytes"]
            assert len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"]
        arrays.append({k: v.cuda() for k, v in load(data).items()})
    torch.cuda.synchronize()
    return payloads, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists")
    torch.cuda.init()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    def used():
        free, total = torch.cuda.mem_get_info()
        return total - free
    before = used()
    start = time.perf_counter()
    loaded = checked_load(args.bundle)
    first = (time.perf_counter() - start) * 1000
    resident = used()
    del loaded
    gc.collect()
    torch.cuda.synchronize()
    after_drop = used()
    for _ in range(5):
        loaded = checked_load(args.bundle)
        del loaded
    trials = []
    for _ in range(30):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loaded = checked_load(args.bundle)
        trials.append((time.perf_counter() - start) * 1000)
        del loaded
    torch.cuda.synchronize()
    manifest = json.loads((args.bundle / "manifest.json").read_bytes())
    report = {"schema_version": 1, "runtime": "pytorch-canonical-adaptation", "device": "cuda",
              "torch_version": torch.__version__, "scope": "read + validate file/tensor SHA256 + copied canonical load + upload + synchronize; destruction excluded",
              "first_load_ms": first, "trials_ms": trials, "warmups": 5,
              "tensor_count": sum(len(r) for r in manifest["tensors"].values()),
              "tensor_bytes": sum(r["bytes"] for records in manifest["tensors"].values() for r in records),
              "device_used_bytes": {"before": before, "loaded": resident, "after_drop": after_drop, "after_trials": used()},
              "device_memory_scope": "global driver snapshots; not process allocation peaks",
              "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cold_disk_cache": False, "context_initialization_timed": False, "model_quality": None}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
