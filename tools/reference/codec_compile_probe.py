"""Optional compiled FP32 output preflight; not full internal-trace qualification."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from codec_bench import checked_model, encode, decode, check_versions
from codec_trace import reference_inputs
from sudoku_artifact import UPSTREAM, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path); p.add_argument("upstream",type=Path); p.add_argument("output",type=Path)
    a=p.parse_args()
    if a.output.exists(): raise ValueError("output exists")
    versions=check_versions()
    if subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM or subprocess.check_output(["git","-C",str(a.upstream),"status","--porcelain"],text=True).strip():
        raise ValueError("unpinned source")
    sys.dont_write_bytecode=True; sys.path.insert(0,str(a.upstream.resolve()))
    from models.vqvae import VQVAE
    root=Path(__file__).resolve().parents[2]
    anchor=json.loads((root/"docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json").read_bytes())
    torch.set_num_threads(1); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    started=time.perf_counter(); rows=[]; error=None
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            model=checked_model(a.bundle,anchor,VQVAE)
            enc=torch.compile(lambda x:encode(model,x),fullgraph=True,dynamic=False)
            dec=torch.compile(lambda x:decode(model,x),fullgraph=True,dynamic=False)
            for name,inputs in reference_inputs(89).items():
                ids=torch.tensor(inputs,dtype=torch.int64,device="cuda")
                expected,snapshots=encode(model,ids); actual,actual_snapshots=enc(ids)
                tensors=[(f"encode_snapshot_{i}",x,y) for i,(x,y) in enumerate(zip(snapshots,actual_snapshots))]
                row={"case":name,"code_disagreements":(expected!=actual).sum().item(),"tensors":[]}
                # Fixed expected codes isolate decode from compiled encode choices.
                e_logits,e_ids,e_snapshots=decode(model,expected)
                c_logits,c_ids,c_snapshots=dec(expected)
                row["output_disagreements"]=(e_ids!=c_ids).sum().item()
                tensors += [("logits",e_logits,c_logits)]+[(f"decode_snapshot_{i}",x,y) for i,(x,y) in enumerate(zip(e_snapshots,c_snapshots))]
                for label,x,y in tensors:
                    delta=(x.double()-y.double()).abs()
                    failed=(delta>5e-4+2e-4*x.double().abs()) | ~torch.isfinite(y)
                    row["tensors"].append({"name":label,"max_abs_error":delta.max().item(),"mismatches":failed.sum().item()})
                rows.append(row)
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc)[:6000]}
    passed=bool(rows) and error is None and all(not r["code_disagreements"] and not r["output_disagreements"] and not any(t["mismatches"] for t in r["tensors"]) for r in rows)
    result={"schema_version":1,"torch":torch.__version__,"upstream_commit":UPSTREAM,"profile":"FP32 fullgraph inductor output preflight",
        "wall_compile_and_preflight_seconds":time.perf_counter()-started,"output_preflight_passed":passed,
        "full_internal_trace_qualification":"pending; this probe cannot establish the full 642-boundary gate",
        "benchmark_baseline_qualified":False,"error":error,"cases":rows,"reference_versions":versions,
        "source_sha256":{name:digest((root/name).read_bytes()) for name in ["tools/reference/codec_compile_probe.py","tools/reference/codec_bench.py","tools/reference/codec_stages.py"]}}
    with a.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False); f.write("\n")
    print("compiled output preflight",passed,"error",error and error["type"])


if __name__=="__main__":main()
