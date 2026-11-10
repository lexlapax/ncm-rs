"""Adapted trusted-input copied MLX load baseline, with exact tensor readback."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import mlx.core as mx
import numpy as np


def load(root, check=False):
    manifest = json.loads((root / "manifest.json").read_bytes())
    payloads = {}
    arrays = []
    for name, record in manifest["files"].items():
        data = (root / name).read_bytes()
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
        if name.endswith(".json"):
            json.loads(data)
        payloads[name] = data
    for name, records in manifest["tensors"].items():
        data = payloads[name]
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8+size])
        assert set(header) - {"__metadata__"} == {r["name"] for r in records}
        for record in records:
            entry = header[record["name"]]
            begin, end = entry["data_offsets"]
            raw = memoryview(data)[8+size+begin:8+size+end]
            assert entry["shape"] == record["shape"] and entry["dtype"] == record["dtype"]
            assert len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"]
            cpu = np.frombuffer(raw, dtype={"F32":"<f4", "I64":"<i8"}[entry["dtype"]]).reshape(entry["shape"])
            array = mx.array(cpu)
            mx.eval(array)
            if check:
                assert np.array(array).tobytes() == raw
            arrays.append(array)
    return payloads, arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path); p.add_argument("output",type=Path)
    args=p.parse_args()
    assert mx.__version__=="0.32.2" and mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    loaded=load(args.bundle,check=True)
    # Match the Rust preflight: a real Metal kernel consumes a loaded weight.
    # Leaf-array eval alone does not initialize the same GPU execution resources.
    weight=next(a for a in loaded[1] if a.dtype==mx.float32)
    kernel=mx.square(weight); mx.eval(kernel); del kernel, weight
    count=len(loaded[1]); del loaded
    mx.clear_cache(); mx.reset_peak_memory()
    before=mx.get_active_memory()
    start=time.perf_counter(); loaded=load(args.bundle)
    first=(time.perf_counter()-start)*1000; resident=mx.get_active_memory()
    del loaded
    after=mx.get_active_memory()
    for _ in range(5):
        loaded=load(args.bundle); del loaded
    trials=[]
    for _ in range(30):
        start=time.perf_counter(); loaded=load(args.bundle)
        trials.append((time.perf_counter()-start)*1000); del loaded
    report={"schema_version":1,"runtime":"python-mlx-0.32.2-adapted","device":"metal",
            "tensor_count":count,"byte_exact_readback":True,"first_load_ms":first,"trials_ms":trials,"warmups":5,
            "scope":"read + integrity verification + copied unified-memory arrays; readback/kernel check excluded",
            "active_memory_bytes":{"before":before,"loaded":resident,"after_drop":after,"after_trials":mx.get_active_memory()},
            "peak_memory_bytes":mx.get_peak_memory(),"cache_memory_bytes":mx.get_cache_memory(),
            "host_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "manifest_sha256":hashlib.sha256((args.bundle/"manifest.json").read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"cold_disk_cache":False,"model_quality":None}
    with args.output.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(f"Python MLX: {count} tensors exact")


if __name__=="__main__":
    main()
