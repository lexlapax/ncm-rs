"""CUDA torch.compile feasibility check; not a statistically qualified timing report."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from operators import prepare, execute, ATOL, RTOL, TORCH_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    assert torch.__version__.split("+")[0] == TORCH_VERSION
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for case in json.loads(args.fixture.read_bytes())["cases"]:
            arrays = prepare(case, "cuda")
            compiled = torch.compile(lambda arrays: execute(case["op"], arrays), fullgraph=True)
            # Capture failures explicitly; never report eager fallback as compiled support.
            try:
                actual = compiled(arrays)
                torch.cuda.synchronize()
                expected = torch.tensor(case["expected"]["data"]).reshape(case["expected"]["shape"])
                torch.testing.assert_close(actual.cpu(), expected, atol=ATOL, rtol=RTOL)
                results.append({"name":case["name"], "passed":True})
            except Exception as error:
                results.append({"name":case["name"], "passed":False, "error":str(error)})
            torch._dynamo.reset()
    with args.output.open("x") as f:
        json.dump({"runtime":"torch.compile-fullgraph-default", "torch_version":torch.__version__,
                   "fixture_sha256":hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                   "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "timings":None, "scope":"compilation capability only; eager report remains timing baseline",
                   "results":results}, f, indent=2)


if __name__ == "__main__":
    main()
