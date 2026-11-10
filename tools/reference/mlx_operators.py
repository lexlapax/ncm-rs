"""Adapted Python MLX operator baseline, checked against the fixed PyTorch oracle."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import mlx.core as mx

ATOL, RTOL = 1e-5, 1e-4


def execute(op, a):
    kind = op["kind"]
    x = a.get("x")
    if kind == "matmul":
        return x @ a["w"]
    if kind == "conv1d":
        return mx.conv1d(x, a["w"].transpose(0, 2, 1), stride=op["stride"], padding=op["padding"])
    if kind == "conv_transpose1d":
        return mx.conv_transpose1d(x, a["w"].transpose(1, 2, 0), stride=op["stride"], padding=op["padding"])
    if kind in ("group_norm", "layer_norm", "rms_norm"):
        if kind == "group_norm":
            b, length, channels = x.shape
            x = x.transpose(0, 2, 1).reshape(b, op["groups"], -1)
        centered = x if kind == "rms_norm" else x - mx.mean(x, axis=-1, keepdims=True)
        y = centered * mx.rsqrt(mx.mean(mx.square(centered), axis=-1, keepdims=True) + op["epsilon"])
        return y.reshape(b, channels, length).transpose(0, 2, 1) if kind == "group_norm" else y
    if kind == "area1d":
        length, source = op["length"], x.shape[1]
        return mx.concatenate([mx.mean(x[:, i*source//length:math.ceil((i+1)*source/length), :], axis=1, keepdims=True) for i in range(length)], axis=1)
    if kind == "linear1d":
        length, source = op["length"], x.shape[1]
        bins = []
        for i in range(length):
            pos = max((i + 0.5) * source / length - 0.5, 0)
            lo = min(math.floor(pos), source - 1)
            hi = min(lo + 1, source - 1)
            weight = pos - lo
            bins.append(x[:, lo:lo+1, :] * (1-weight) + x[:, hi:hi+1, :] * weight)
        return mx.concatenate(bins, axis=1)
    if kind == "attention":
        scores = (a["q"] @ mx.swapaxes(a["k"], -1, -2)) * (1/math.sqrt(a["q"].shape[-1])) + a["mask"]
        return mx.softmax(scores, axis=-1, precise=True) @ a["v"]
    if kind == "rope":
        half = x.shape[-1] // 2
        return x*a["cos"] + mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)*a["sin"]
    if kind == "f16_roundtrip":
        half = x.astype(mx.float16)
        mx.eval(half)
        return half.astype(mx.float32)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if mx.__version__ != "0.32.2" or not mx.metal.is_available():
        raise RuntimeError("requires MLX 0.32.2 and real Metal; no fallback")
    mx.set_default_device(mx.gpu)
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    assert fixture["schema_version"] == 1 and fixture["upstream_commit"] == "c70274273f489abc73832b2b98584a8483b6b203"
    mx.reset_peak_memory()
    results = []
    for case in fixture["cases"]:
        a = {name:mx.array(t["data"], dtype=mx.float32).reshape(t["shape"]) for name,t in case["inputs"].items()}
        if "mask" in a:
            a["mask"] = mx.where(a["mask"] == 1, 0., -float("inf"))
        mx.eval(*a.values())
        start = time.perf_counter()
        output = execute(case["op"], a)
        mx.eval(output)
        first = (time.perf_counter()-start)*1000
        assert list(output.shape) == case["expected"]["shape"]
        actual = output.reshape(-1).tolist()
        expected = case["expected"]["data"]
        assert all(math.isfinite(x) for x in actual)
        errors = [abs(x-y) for x,y in zip(actual, expected)]
        mismatches = sum(e > ATOL+RTOL*abs(y) for e,y in zip(errors,expected))
        del output
        for _ in range(5):
            output=execute(case["op"],a); mx.eval(output); del output
        trials=[]
        for _ in range(30):
            start=time.perf_counter(); output=execute(case["op"],a); mx.eval(output)
            trials.append((time.perf_counter()-start)*1000); del output
        results.append({"name":case["name"],"first_execution_ms":first,"trials_ms":trials,
                        "errors":{"passed":mismatches==0,"mismatches":mismatches,"elements":len(errors),
                                  "max_abs_error":max(errors),"mean_abs_error":sum(errors)/len(errors)}})
    report={"schema_version":1,"runtime":"python-mlx-0.32.2-adapted","device":"metal",
            "fixture_sha256":hashlib.sha256(raw).hexdigest(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dtype":"float32","warmups":5,"trials":30,"atol":ATOL,"rtol":RTOL,
            "scope":"graph construction + synchronous eval; uploads/readback excluded; FP16 cast explicitly materialized",
            "peak_memory_bytes":mx.get_peak_memory(),"model_quality":None,
            "passed":all(r["errors"]["passed"] for r in results),"results":results}
    with args.output.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
    if not report["passed"]:
        raise RuntimeError("MLX reference parity failed")
    print("Python Metal: all 18 operator cases pass")


if __name__ == "__main__":
    main()
