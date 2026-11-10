"""Fresh-process CUDA lifecycle diagnostic; not steady-state acceptance timing."""
import json
from pathlib import Path
import subprocess
import sys
import time
from codec_bench import (ROOT, UPSTREAM, call, checked_model, check_versions, ids_hash,
                         torch, sdpa_kernel, SDPBackend)


def emit(phase, start, digest=None):
    print(json.dumps({"schema_version":1,"phase":phase,"internal_ms":(time.perf_counter()-start)*1000,
                      "output_ids_sha256":digest}), flush=True)


def main():
    bundle, upstream = map(Path, sys.argv[1:])
    check_versions()
    if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM or subprocess.check_output(["git","-C",str(upstream),"status","--porcelain"],text=True).strip():
        raise ValueError("unpinned/dirty source")
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(upstream.resolve()))
    from models.vqvae import VQVAE
    torch.set_num_threads(1); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    start=time.perf_counter(); torch.cuda.init(); torch.cuda.synchronize(); emit("context",start)
    anchor=json.loads((ROOT/"docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json").read_bytes())
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        start=time.perf_counter(); model=checked_model(bundle,anchor,VQVAE); emit("load",start)
        ids=[]
        for row in range(9):
            if row: ids.append(10)
            ids.extend((row*3+row//3+col)%9+1 for col in range(9))
        start=time.perf_counter(); result=call(model,"reconstruct",ids,[],1,None)
        if len(result[1])!=411 or result[1][322:]!=ids: raise ValueError("incorrect solved-board output")
        emit("execute",start,ids_hash(result[1]))
        start=time.perf_counter(); del result,model; torch.cuda.synchronize(); emit("drop",start)


if __name__=="__main__": main()
