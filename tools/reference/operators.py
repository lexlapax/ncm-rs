"""Phase 1 synthetic operator oracle and eager PyTorch microprobe (no NCM model)."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

UPSTREAM = "c70274273f489abc73832b2b98584a8483b6b203"
TORCH_VERSION = "2.15.0"
ATOL, RTOL = 1e-5, 1e-4


def tensor(shape, offset=0):
    # Binary fractions make the inputs reproducible without a shared RNG.
    size = math.prod(shape)
    values = [((i * 7 + offset) % 31 - 15) / 16 for i in range(size)]
    return {"shape": shape, "data": values}


def serialize(value):
    return {"shape": list(value.shape), "data": value.flatten().tolist()}


def prepare(case, device):
    arrays = {
        name: torch.tensor(value["data"], dtype=torch.float32, device=device).reshape(value["shape"])
        for name, value in case["inputs"].items()
    }
    if "mask" in arrays:
        arrays["mask"] = torch.where(arrays["mask"] == 1, 0.0, -torch.inf)
    return arrays


def execute(op, arrays):
    kind = op["kind"]
    x = arrays.get("x")
    if kind == "matmul":
        return x @ arrays["w"]
    if kind == "conv1d":
        return F.conv1d(x.transpose(1, 2).contiguous(), arrays["w"],
                        stride=op["stride"], padding=op["padding"]).transpose(1, 2).contiguous()
    if kind == "conv_transpose1d":
        return F.conv_transpose1d(x.transpose(1, 2).contiguous(), arrays["w"],
                                 stride=op["stride"], padding=op["padding"]).transpose(1, 2).contiguous()
    if kind == "group_norm":
        return F.group_norm(x.transpose(1, 2).contiguous(), op["groups"],
                            eps=op["epsilon"]).transpose(1, 2).contiguous()
    if kind == "layer_norm":
        return F.layer_norm(x, [x.shape[-1]], eps=op["epsilon"])
    if kind == "rms_norm":
        return x * (x.square().mean(-1, keepdim=True) + op["epsilon"]).rsqrt()
    if kind in ("area1d", "linear1d"):
        mode = "area" if kind == "area1d" else "linear"
        kwargs = {} if mode == "area" else {"align_corners": False}
        return F.interpolate(x.transpose(1, 2), size=op["length"], mode=mode,
                             **kwargs).transpose(1, 2).contiguous()
    if kind == "attention":
        return F.scaled_dot_product_attention(arrays["q"], arrays["k"], arrays["v"],
                                             attn_mask=arrays["mask"], dropout_p=0.0)
    if kind == "rope":
        first, second = x.chunk(2, dim=-1)
        return x * arrays["cos"] + torch.cat([-second, first], dim=-1) * arrays["sin"]
    if kind == "f16_roundtrip":
        return x.half().float()
    raise ValueError(f"unknown operation: {kind}")


def generate():
    cases = []

    def add(name, kind, inputs, **params):
        op = {"kind": kind, **params}
        case = {"name": name, "op": op, "inputs": inputs}
        case["expected"] = serialize(execute(op, prepare(case, "cpu")))
        cases.append(case)

    add("matmul_asymmetric", "matmul", {"x": tensor([8, 16]), "w": tensor([16, 12], 3)})
    add("conv1d_layout", "conv1d", {"x": tensor([2, 7, 2]), "w": tensor([3, 2, 3], 5)}, stride=1, padding=1)
    add("conv_transpose1d_layout", "conv_transpose1d",
        {"x": tensor([2, 3, 2]), "w": tensor([2, 3, 3], 9)}, stride=2, padding=1)
    add("conv_transpose1d_prime89", "conv_transpose1d",
        {"x": tensor([1, 1, 2]), "w": tensor([2, 3, 89], 2)}, stride=89, padding=0)
    add("group_norm_sequence_axes", "group_norm", {"x": tensor([2, 7, 4])}, groups=2, epsilon=1e-5)
    add("layer_norm", "layer_norm", {"x": tensor([2, 7, 4])}, epsilon=1e-6)
    add("rms_norm", "rms_norm", {"x": tensor([2, 7, 4])}, epsilon=1e-6)
    for kind, source, target in [("area1d", 7, 4), ("area1d", 89, 13),
                                 ("area1d", 1, 1), ("area1d", 3, 7),
                                 ("linear1d", 3, 7), ("linear1d", 1, 9),
                                 ("linear1d", 7, 3), ("linear1d", 7, 7)]:
        add(f"{kind}_{source}_to_{target}", kind, {"x": tensor([2, source, 3])}, length=target)
    # Two prefix queries, three target queries, prefix key 1 padded out.
    mask = [int(k != 1 and (q >= 2 or k < 2)) for q in range(5) for k in range(5)]
    add("prefix_attention_padded", "attention", {
        "q": tensor([1, 2, 5, 4]), "k": tensor([1, 2, 5, 4], 4),
        "v": tensor([1, 2, 5, 4], 8), "mask": {"shape": [1, 1, 5, 5], "data": mask},
    })
    # Positions reset after a three-position prefix; two target levels are included.
    positions = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=torch.float32)
    frequencies = positions[:, None] * (1.0 / (10000 ** (torch.arange(2).float() / 2)))
    cos = torch.cat([frequencies.cos()] * 2, dim=-1)
    sin = torch.cat([frequencies.sin()] * 2, dim=-1)
    add("rope_split_half_level_resets", "rope", {
        "x": tensor([1, 2, 9, 4]), "cos": serialize(cos), "sin": serialize(sin),
    })
    add("f16_roundtrip", "f16_roundtrip", {"x": {
        "shape": [2, 3], "data": [0.12345, 1.0003, -2.0006, 65504.0, 0.000061035, 0.0],
    }})
    return {"schema_version": 1, "upstream_commit": UPSTREAM,
            "torch_version": torch.__version__, "dtype": "float32", "cases": cases}


def run(fixture, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    results = []
    for case in fixture["cases"]:
        arrays = prepare(case, device)
        sync()
        start = time.perf_counter()
        output = execute(case["op"], arrays)
        sync()
        first_ms = (time.perf_counter() - start) * 1000
        actual = output.cpu().flatten().tolist()
        reference = case["expected"]["data"]
        if list(output.shape) != case["expected"]["shape"] or len(actual) != len(reference):
            raise ValueError("output shape mismatch")
        errors = [abs(a - b) for a, b in zip(actual, reference)]
        if not all(math.isfinite(x) for x in actual):
            raise ValueError("nonfinite output")
        mismatches = sum(e > ATOL + RTOL * abs(b) for e, b in zip(errors, reference))
        del output
        for _ in range(5):
            output = execute(case["op"], arrays)
            sync()
            del output
        trials = []
        for _ in range(30):
            sync()
            start = time.perf_counter()
            output = execute(case["op"], arrays)
            sync()
            trials.append((time.perf_counter() - start) * 1000)
            del output
        results.append({"name": case["name"], "first_execution_ms": first_ms,
                        "trials_ms": trials, "errors": {"passed": mismatches == 0,
                            "max_abs_error": max(errors), "mean_abs_error": sum(errors) / len(errors),
                            "mismatches": mismatches, "elements": len(errors)}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "run"])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise RuntimeError(f"oracle requires torch {TORCH_VERSION}; got {torch.__version__}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        if args.action == "generate":
            with args.fixture.open("x") as output:
                json.dump(generate(), output, indent=2, allow_nan=False)
                output.write("\n")
            return
        if args.output is None:
            parser.error("run requires --output")
        data = args.fixture.read_bytes()
        fixture = json.loads(data)
        if (fixture["schema_version"] != 1 or fixture["upstream_commit"] != UPSTREAM
                or fixture["dtype"] != "float32"):
            raise ValueError("fixture provenance/schema mismatch")
        results = run(fixture, args.device)
    report = {
        "schema_version": 1, "runtime": "pytorch-eager-math-sdpa", "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.machine(),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(data).hexdigest(), "upstream_commit": UPSTREAM,
        "dtype": "float32", "tf32": False, "warmups": 5, "trials": 30,
        "scope": "operator dispatch + device synchronization; uploads/readback excluded",
        "atol": ATOL, "rtol": RTOL, "model_quality": None, "peak_memory": None,
        "passed": all(r["errors"]["passed"] for r in results), "results": results,
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    if not report["passed"]:
        raise RuntimeError("operator parity failed; see report")
    print(f"{args.device}: {len(results)} operator cases passed; report {args.output}")


if __name__ == "__main__":
    main()
