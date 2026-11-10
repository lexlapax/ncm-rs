"""Capture the adapted Python MLX codec against authenticated CUDA input fixtures."""
import argparse
import json
from pathlib import Path
import subprocess
import mlx.core as mx
import numpy as np
from mlx_codec import Codec, prepare_device
from mlx_codec_assets import ROOT, UPSTREAM, digest, tensors, save_tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path);p.add_argument("reference",type=Path);p.add_argument("upstream",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError("output exists")
    if subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM or subprocess.check_output(["git","-C",str(a.upstream),"status","--porcelain"],text=True).strip():raise ValueError("unpinned/dirty upstream")
    report_bytes=(a.reference/"report.json").read_bytes()
    if report_bytes!=(ROOT/"docs/results/2026-11-10-phase2-oracle/raw/sudoku-cuda.json").read_bytes():raise ValueError("untrusted reference report")
    report=json.loads(report_bytes);raw=(a.reference/"traces.safetensors").read_bytes()
    if digest(raw)!=report["trace_sha256"]:raise ValueError("reference trace hash mismatch")
    reference=tensors(raw);config_bytes=(a.reference/"config.json").read_bytes()
    if digest(config_bytes)!=report["config_sha256"]:raise ValueError("reference config mismatch")
    prepare_device();codec=Codec.load(a.bundle)
    if codec.config!=json.loads(config_bytes):raise ValueError("runtime config differs")
    weights={name:{"shape":list(value.shape),"sha256":digest(np.array(value).astype("<f4").tobytes())} for name,value in codec.w.items()}
    if weights!=report["weights"]:raise ValueError("runtime weight readback differs")
    # Explicit first-index/signed-zero regression, independent of model scores.
    tie=np.zeros((2,1025),dtype=np.float32);tie[0,3]=tie[0,1024]=-7.;tie[0,27]=tie[0,1001]=9.;tie[1,1]=-0.
    tie=mx.array(tie)
    if mx.argmin(tie,axis=-1).tolist()!=[3,0] or mx.argmax(tie,axis=-1).tolist()!=[27,0]:raise ValueError("selection tie order differs")
    arrays={}
    for case in report["cases"]:
        name=case["name"];trace={}
        tokens=reference[name+"/input.ids"];supplied=reference[name+"/supplied.codes"]
        if np.any(tokens<0) or np.any(tokens>=12) or np.any(supplied<0) or np.any(supplied>=16):raise ValueError("invalid reference IDs")
        ids=mx.array(tokens.astype(np.uint32));forced=mx.array(supplied.astype(np.uint32))
        trace["input.ids"]=ids;trace["supplied.codes"]=forced
        codes,encoded=codec.encode(ids,trace)
        logits,output,decoded=codec.decode(codes,trace)
        f_logits,f_output,f_decoded=codec.decode(forced,trace,"forced")
        mx.eval(*trace.values())
        for phase,snapshots in (("encode",encoded),("decode",decoded),("forced",f_decoded)):
            for level,value in enumerate(snapshots):
                if not np.array_equal(np.array(value),np.array(trace[f"{phase}.level_{level:02}.cumulative"])):raise ValueError("snapshot changed")
        plain_codes,plain_encoded=codec.encode(ids)
        plain_logits,plain_output,plain_decoded=codec.decode(plain_codes)
        pf_logits,pf_output,pf_decoded=codec.decode(forced)
        for x,y in ((codes,plain_codes),(logits,plain_logits),(output,plain_output),(f_logits,pf_logits),(f_output,pf_output)):
            if not np.array_equal(np.array(x),np.array(y)):raise ValueError("uninstrumented output differs")
        for key,value in trace.items():
            array=np.array(value)
            if not np.isfinite(array).all():raise ValueError("nonfinite observation")
            arrays[name+"/"+key]=array
    if set(arrays)!=set(reference):raise ValueError("capture inventory differs")
    a.output.mkdir(parents=True)
    save_tensors(a.output/"traces.safetensors",arrays)
    (a.output/"config.json").write_bytes(config_bytes)
    sources=["tools/reference/mlx_codec.py","tools/reference/mlx_codec_assets.py","tools/reference/mlx_codec_trace.py","tools/reference/mlx_operators.py","tools/reference/requirements-macos.lock"]
    result={"schema_version":1,"profile":"sudoku","runtime":"python-mlx-adapted","device":"metal","upstream_commit":UPSTREAM,
        "policy":report["policy"],"config_sha256":report["config_sha256"],"weights":weights,
        "trace_sha256":digest((a.output/"traces.safetensors").read_bytes()),"trace_count":len(arrays),"reference_report_sha256":digest(report_bytes),
        "source_sha256":{name:digest((ROOT/name).read_bytes()) for name in sources},
        "upstream_source_sha256":{name:digest((a.upstream/name).read_bytes()) for name in ("models/vae.py","models/vqvae.py","models/quant.py")},
        "adaptations":["independent Python MLX FP32 port of pinned Sudoku codec only, no autograd/training buffers",
            "explicit bidirectional math attention and exact-erf GELU; composed area/half-pixel resampling",
            "native FP32 projection for full-span convolution; no numerical budget change",
            "independent cumulative snapshots; omit unused encode z_levels, no broken reconstruct helper"],
        "uninstrumented_matches_traced":True,"first_index_ties_pass":True,"performance":None}
    with (a.output/"report.json").open("x") as file:json.dump(result,file,indent=2,allow_nan=False);file.write("\n")
    print(len(arrays),"adapted Python MLX traces captured; independent qualification still required")


if __name__=="__main__":main()
