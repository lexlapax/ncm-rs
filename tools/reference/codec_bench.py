"""Pinned eager CUDA codec timings with explicitly snapshot-safe decode outputs."""
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load
from torch.nn.attention import SDPBackend, sdpa_kernel
from codec_trace import reference_inputs
from sudoku_artifact import UPSTREAM, digest
from codec_stages import profile

ROOT = Path(__file__).resolve().parents[2]


def check_versions():
    versions = {}
    for line in (ROOT/"tools/reference/requirements-linux.lock").read_text().splitlines():
        package,expected = line.split("==")
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected: raise ValueError(f"unpinned {package}: {actual}")
        versions[package] = actual
    return versions


def checked_model(root, anchor, cls):
    manifest = json.loads((root/"manifest.json").read_bytes())
    if manifest != anchor:
        raise ValueError("artifact differs from pinned inventory")
    payloads = {}
    for name, record in manifest["files"].items():
        data = (root/name).read_bytes()
        if len(data) != record["bytes"] or digest(data) != record["sha256"]:
            raise ValueError("artifact file mismatch")
        if name.endswith(".json"):
            json.loads(data)
        payloads[name] = data
    for name, records in manifest["tensors"].items():
        data = payloads[name]
        size = int.from_bytes(data[:8],"little")
        header = json.loads(data[8:8+size])
        if set(header)-{"__metadata__"} != {r["name"] for r in records}:
            raise ValueError("tensor inventory differs")
        for record in records:
            entry = header[record["name"]]
            begin,end = entry["data_offsets"]
            raw = memoryview(data)[8+size+begin:8+size+end]
            if entry["shape"] != record["shape"] or entry["dtype"] != record["dtype"] or len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
                raise ValueError("tensor layout/hash differs")
    config = json.loads(payloads["config.json"])["tokenizer"]["vqvae"]
    with contextlib.redirect_stdout(io.StringIO()):
        codec = cls(**OmegaConf.create(config)).eval()
    state = load(payloads["codec.safetensors"])
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("invalid runtime weight")
    state.update(load(payloads["training.safetensors"]))
    codec.load_state_dict(state,strict=True)
    codec = codec.cuda()
    torch.cuda.synchronize()
    return codec


def encode(codec, ids, timing=None):
    # Preserve the original encode helper, including its unused z_levels work.
    if timing is not None: timing.start("encoder_pre_quant")
    latent = codec.encode_z(ids)
    if timing is not None: timing.end("encoder_pre_quant")
    return codec.vq.z_to_idx(latent,return_zhats=True)


def decode(codec, ids, timing=None):
    # Reference fix: idx_to_zhats aliases its accumulator. Use out-of-place sums
    # and retain every cumulative latent, matching the native public API.
    value = torch.zeros(ids.shape[0],codec.context_length,codec.embed_dim,device=ids.device)
    snapshots = []
    offset = 0
    for level,length in enumerate(codec.vq.levels):
        q = codec.vq.embeddings[level](ids[:,offset:offset+length])
        offset += length
        if level+1 != len(codec.vq.levels):
            q = codec.vq._up_to_full(q,level,codec.context_length)
        q = codec.vq._apply_quant_resi(q,level)
        value = value+q
        snapshots.append(value)
    if timing is not None: timing.start("decoder_logits")
    logits = codec.decoder(value)
    if timing is not None: timing.end("decoder_logits"); timing.start("output_selection")
    output = logits.argmax(-1)
    if timing is not None: timing.end("output_selection")
    return logits,output,snapshots


def host():
    status = Path("/proc/self/status").read_text()
    return {key: int(next(line for line in status.splitlines() if line.startswith(key+":" )).split()[1])*1024 for key in ("VmRSS","VmHWM")}


def memory():
    free,total = torch.cuda.mem_get_info()
    return {"allocated_current":torch.cuda.memory_allocated(),"allocated_peak":torch.cuda.max_memory_allocated(),
            "reserved_current":torch.cuda.memory_reserved(),"reserved_peak":torch.cuda.max_memory_reserved(),
            "device_global_used":total-free}


def ids_hash(ids):
    import struct
    return hashlib.sha256(struct.pack("<"+"I"*len(ids),*ids)).hexdigest()


def call(codec, scope, tokens, codes, batch, latent):
    if scope == "resident_decoder_block":
        out = codec.decoder.middle_block(latent)
        torch.cuda.synchronize()
        return [out],[],None
    if scope in ("encode","reconstruct"):
        ids = torch.tensor(tokens,dtype=torch.int64,device="cuda").reshape(batch,89)
        selected,snapshots = encode(codec,ids)
        if scope == "encode":
            torch.cuda.synchronize()
            return snapshots,selected.cpu().flatten().tolist(),None
        logits,output,decoded_snapshots = decode(codec,selected)
        torch.cuda.synchronize()
        return snapshots+decoded_snapshots+[logits],selected.cpu().flatten().tolist()+output.cpu().flatten().tolist(),logits
    ids = torch.tensor(codes,dtype=torch.int64,device="cuda").reshape(batch,-1)
    logits,output,snapshots = decode(codec,ids)
    torch.cuda.synchronize()
    return snapshots+[logits],output.cpu().flatten().tolist(),logits


def timed(function):
    torch.cuda.synchronize()
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    begin = time.perf_counter()
    output = function()
    end.record(); end.synchronize()
    return output,(time.perf_counter()-begin)*1000,start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle",type=Path); parser.add_argument("upstream",type=Path)
    parser.add_argument("workload",type=Path); parser.add_argument("output",type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("output exists")
    versions = check_versions()
    if subprocess.check_output(["git","-C",str(args.upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM or subprocess.check_output(["git","-C",str(args.upstream),"status","--porcelain"],text=True).strip():
        raise ValueError("unpinned/dirty source")
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    workload = json.loads(args.workload.read_bytes())
    if workload != json.loads((ROOT/"tests/fixtures/codec-bench-v1.json").read_bytes()):
        raise ValueError("unsupported protocol")
    anchor = json.loads((ROOT/"docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json").read_bytes())
    torch.set_num_threads(1); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.cuda.init(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        before = memory(); begin = time.perf_counter(); model = checked_model(args.bundle,anchor,VQVAE)
        first_load = (time.perf_counter()-begin)*1000; loaded = memory(); del model; gc.collect()
        loads=[]
        for trial in range(35):
            begin = time.perf_counter(); model = checked_model(args.bundle,anchor,VQVAE)
            elapsed = (time.perf_counter()-begin)*1000
            if trial>=5: loads.append(elapsed)
            del model
        torch.cuda.synchronize(); load_memory={"before":before,"first_loaded":loaded,"after_trials":memory(),"host":host()}
        codec = checked_model(args.bundle,anchor,VQVAE)
        base = reference_inputs(89)["solved_batch2"]
        def data(batch):
            tokens = sum((base[i%2] for i in range(batch)),[])
            codes = [((i*length+p+3*level)%16) for i in range(batch) for level,length in enumerate(codec.vq.levels) for p in range(length)]
            return tokens,codes
        tokens,codes = data(1)
        cold,cold_ms,cold_span = timed(lambda:call(codec,"reconstruct",tokens,codes,1,None)); del cold
        rows=[]; stage_rows=[]
        for batch in workload["batches"]:
            tokens,codes = data(batch)
            encoded,snapshots = encode(codec,torch.tensor(tokens,dtype=torch.int64,device="cuda").reshape(batch,89))
            latent = snapshots[-1]; del encoded,snapshots
            # Corrected decode must equal the unmodified pinned final decode exactly.
            supplied = torch.tensor(codes,dtype=torch.int64,device="cuda").reshape(batch,-1)
            corrected,_,snapshots = decode(codec,supplied)
            if not torch.equal(corrected,codec.decode(supplied)):
                raise ValueError("snapshot-safe reference differs from pinned decoder")
            del corrected,snapshots,supplied
            for scope in workload["scopes"]:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                first,first_ms,first_span = timed(lambda:call(codec,scope,tokens,codes,batch,latent))
                floats,output,logits = first
                expected_hash = ids_hash(output)
                first_logits = logits.cpu().flatten().tolist() if logits is not None else None
                float_elements = sum(x.numel() for x in floats); output_count=len(output)
                del first,floats,output,logits
                wall,spans=[],[]
                for trial in range(35):
                    value,ms,span = timed(lambda:call(codec,scope,tokens,codes,batch,latent))
                    if ids_hash(value[1])!=expected_hash: raise ValueError("nonrepeatable output IDs")
                    if trial>=5: wall.append(ms); spans.append(span)
                    del value
                torch.cuda.synchronize()
                rows.append({"batch":batch,"scope":scope,"first_ms":first_ms,"first_cuda_stream_span_ms":first_span,
                    "trials_ms":wall,"cuda_stream_spans_ms":spans,"output_ids_sha256":expected_hash,"output_ids_count":output_count,
                    "output_float_elements":float_elements,"input_ids_sha256":ids_hash(tokens),"supplied_codes_sha256":ids_hash(codes),
                    "first_logits":first_logits,"allocator_bytes":memory(),"host":host()})
            stage_rows.append({"batch":batch,"cuda_stream_spans_ms":profile(codec,tokens,batch,encode,decode)})
        report={"schema_version":1,"runtime":"pytorch-pinned-eager-snapshot-fix","torch":torch.__version__,"device":"cuda",
            "workload_sha256":digest(args.workload.read_bytes()),"manifest_sha256":digest((args.bundle/"manifest.json").read_bytes()),
            "source_sha256":{"benchmark":digest(Path(__file__).read_bytes()),"inputs_helper":digest((ROOT/"tools/reference/codec_trace.py").read_bytes()),"stages":digest((ROOT/"tools/reference/codec_stages.py").read_bytes())},
            "upstream_commit":UPSTREAM,"reference_versions":versions,"precision":"fp32","index_dtype":"i64","warmups":5,"trials":30,
            "first_load_ms":first_load,"load_trials_ms":loads,"load_memory":load_memory,"measurements":rows,"instrumented_stages":stage_rows,
            "first_execution_ms":cold_ms,"first_execution_cuda_stream_span_ms":cold_span,"first_execution_scope":"batch-one reconstruction before component preparation",
            "host_final":host(),"cold_disk_cache":False,"context_initialization_timed":False,
            "reference_notes":["encode uses unmodified helper, which also computes unused resampled z_levels",
                "decode uses snapshot-safe out-of-place accumulator; outputs checked bit-exact against unmodified decoder",
                "upstream module retains training buffers on GPU; Rust runtime uploads only 58 inference tensors"],
            "compile_profile":"eager math SDPA, TF32 off; compiled profile assessed separately"}
    with args.output.open("x") as file:
        json.dump(report,file,indent=2,allow_nan=False); file.write("\n")


if __name__=="__main__":
    main()
