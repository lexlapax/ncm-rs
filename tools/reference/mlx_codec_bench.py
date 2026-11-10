"""Matched Metal codec timings using the separately qualified Python MLX port."""
import argparse
import json
from pathlib import Path
import struct
import time
import mlx.core as mx
from mlx_codec import Codec,prepare_device
from mlx_codec_assets import ROOT,digest
from text_tokenizer_bench import host


def check_baseline(root):
    raw=(root/"report.json").read_bytes();report=json.loads(raw)
    quality_raw=(root/"qualification.json").read_bytes();quality=json.loads(quality_raw)
    if report["runtime"]!="python-mlx-adapted" or quality.get("benchmark_baseline_qualified") is not True or quality["candidate_report_sha256"]!=digest(raw) or quality["right_trace_sha256"]!=report["trace_sha256"]:
        raise ValueError("unqualified baseline")
    if quality["qualification_script_sha256"]!=digest((ROOT/"tools/reference/qualify_mlx_reference.py").read_bytes()):raise ValueError("qualification tool changed")
    for name,expected in report["source_sha256"].items():
        if digest((ROOT/name).read_bytes())!=expected:raise ValueError(f"qualified source changed: {name}")
    return digest(quality_raw)


def memory():
    return {"active":mx.get_active_memory(),"peak":mx.get_peak_memory(),"cache":mx.get_cache_memory()}


def data(batch,levels):
    tokens=[]
    for i in range(batch):
        for row in range(9):
            if row:tokens.append(10)
            tokens.extend((row*3+row//3+col+i%2)%9+1 for col in range(9))
    codes=[(i*length+p+3*level)%16 for i in range(batch) for level,length in enumerate(levels) for p in range(length)]
    return tokens,codes


def ids_hash(ids):
    return digest(struct.pack("<"+"I"*len(ids),*ids))


def call(codec,scope,tokens,codes,batch,latent):
    if scope=="resident_decoder_block":
        value=codec.block(latent);mx.eval(value);return [value],[],None
    if scope in ("encode","reconstruct"):
        ids=mx.array(tokens,dtype=mx.uint32).reshape(batch,89)
        selected,snapshots=codec.encode(ids)
        if scope=="encode":
            mx.eval(*snapshots);return snapshots,selected.reshape(-1).tolist(),None
        logits,output,decoded=codec.decode(selected)
        floats=snapshots+decoded+[logits];mx.eval(*floats)
        return floats,selected.reshape(-1).tolist()+output.reshape(-1).tolist(),logits
    logits,output,snapshots=codec.decode(mx.array(codes,dtype=mx.uint32).reshape(batch,-1))
    floats=snapshots+[logits];mx.eval(*floats)
    return floats,output.reshape(-1).tolist(),logits


def timed(function):
    mx.synchronize();start=time.perf_counter();output=function()
    return output,(time.perf_counter()-start)*1000


class Stages(dict):
    def __init__(self):
        super().__init__();self.start=None;self.values={}

    def begin(self,name):
        if self.start is not None:raise ValueError("nested stage")
        self.start=(name,time.perf_counter())

    def end(self,name):
        if self.start is None or self.start[0]!=name:raise ValueError("mismatched stage")
        self.values[name]=(time.perf_counter()-self.start[1])*1000;self.start=None

    def __setitem__(self,name,value):
        if name=="encode.encoder.input":value.reshape(-1).tolist();self.begin("encoder_pre_quant")
        if name=="encode.pre_quant":mx.eval(value);self.end("encoder_pre_quant")
        if name=="decode.decoder.input":mx.eval(value);self.begin("decoder_logits")
        if name=="decode.logits":mx.eval(value);self.end("decoder_logits");self.begin("output_selection_readback")
        if name=="decode.ids":value.reshape(-1).tolist();self.end("output_selection_readback")
        if name.startswith("encode.level_") and name.endswith(".residual_before"):
            mx.eval(value);self.begin(name.removesuffix(".residual_before"))
        if name.startswith("encode.level_") and name.endswith(".contribution"):
            mx.eval(value);self.end(name.removesuffix(".contribution"))


def profile(codec,tokens,batch):
    trials={}
    for trial in range(35):
        ids=mx.array(tokens,dtype=mx.uint32).reshape(batch,89);stages=Stages()
        codes,encoded=codec.encode(ids,stages);output=codec.decode(codes,stages)
        if stages.start is not None or len(stages.values)!=15:raise ValueError("incomplete stages")
        if trial>=5:
            for name,ms in stages.values.items():trials.setdefault(name,[]).append(ms)
        del codes,encoded,output,ids,stages
    return trials


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path);p.add_argument("workload",type=Path);p.add_argument("qualified_baseline",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError("output exists")
    qualified=check_baseline(a.qualified_baseline);raw=a.workload.read_bytes();workload=json.loads(raw)
    if workload!=json.loads((ROOT/"tests/fixtures/codec-bench-v1.json").read_bytes()):raise ValueError("unpinned workload")
    prepare_device();mx.clear_cache();mx.reset_peak_memory();before=memory()
    codec,first_load=timed(lambda:Codec.load(a.bundle));loaded=memory();del codec;dropped=memory()
    loads=[]
    for trial in range(35):
        codec,ms=timed(lambda:Codec.load(a.bundle))
        if trial>=5:loads.append(ms)
        del codec
    load_memory={"before":before,"first_loaded":loaded,"first_dropped":dropped,"after_trials":memory(),"host":host()}
    codec=Codec.load(a.bundle);tokens,codes=data(1,codec.levels)
    cold,first_execution=timed(lambda:call(codec,"reconstruct",tokens,codes,1,None));del cold
    rows=[];stages=[]
    for batch in workload["batches"]:
        tokens,codes=data(batch,codec.levels)
        encoded,snapshots=codec.encode(mx.array(tokens,dtype=mx.uint32).reshape(batch,89))
        latent=snapshots[-1];mx.eval(latent);del encoded,snapshots
        for scope in workload["scopes"]:
            mx.reset_peak_memory()
            first,first_ms=timed(lambda:call(codec,scope,tokens,codes,batch,latent))
            floats,output,logits=first
            hashed=ids_hash(output);count=len(output);elements=sum(x.size for x in floats)
            first_logits=logits.reshape(-1).tolist() if logits is not None else None
            del first,floats,output,logits
            wall=[]
            for trial in range(35):
                value,ms=timed(lambda:call(codec,scope,tokens,codes,batch,latent))
                if ids_hash(value[1])!=hashed:raise ValueError("nonrepeatable IDs")
                if trial>=5:wall.append(ms)
                del value
            rows.append({"batch":batch,"scope":scope,"first_ms":first_ms,"trials_ms":wall,"output_ids_sha256":hashed,"output_ids_count":count,"output_float_elements":elements,
                "input_ids_sha256":ids_hash(tokens),"supplied_codes_sha256":ids_hash(codes),"first_logits":first_logits,"allocator_bytes":memory(),"host":host()})
        stages.append({"batch":batch,"synchronized_stage_ms":profile(codec,tokens,batch)})
    paths=["tools/reference/mlx_codec_bench.py","tools/reference/mlx_codec.py","tools/reference/mlx_codec_assets.py","tools/reference/mlx_operators.py","tools/reference/text_tokenizer_bench.py","tools/reference/requirements-macos.lock"]
    report={"schema_version":1,"runtime":"python-mlx-adapted","device":"metal","precision":"fp32","index_dtype":"u32","warmups":5,"trials":30,
        "workload_sha256":digest(raw),"manifest_sha256":digest((a.bundle/"manifest.json").read_bytes()),"source_sha256":{name:digest((ROOT/name).read_bytes()) for name in paths},
        "baseline_qualification_sha256":qualified,"first_load_ms":first_load,"load_trials_ms":loads,"load_memory":load_memory,"first_execution_ms":first_execution,
        "measurements":rows,"instrumented_stages":stages,"host_final":host(),"cold_disk_cache":False,"context_initialization_timed":False,
        "timing_scope":"synchronous graph construction and evaluation; resident middle block excludes host copies, not kernel-only time; output destruction excluded",
        "stage_scope":"separate synchronized barriers perturb execution; includes final selection host readback; not additive headline timings",
        "memory_scope":"MLX active/peak/cache unified-memory allocator; process RSS via ps and HWM via Darwin getrusage; not separate VRAM",
        "compile_profile":"eager MLX compositions; no graph compilation; adapted baseline, not upstream Apple execution"}
    with a.output.open("x") as file:json.dump(report,file,indent=2,allow_nan=False);file.write("\n")


if __name__=="__main__":main()
