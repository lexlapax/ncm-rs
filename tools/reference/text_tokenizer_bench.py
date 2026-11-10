"""Pinned original tokenizer wrappers; safe source loading and matched scalar trials."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from convert_tokenizers import ROOT, UPSTREAM, VERSIONS, SPECIAL, data_pickle, validate_bpe, digest


def host():
    lines = Path("/proc/self/status").read_text().splitlines()
    return {key:int(next(line for line in lines if line.startswith(key+":")).split()[1])*1024 for key in ("VmRSS","VmHWM")}


def timed(function):
    start = time.perf_counter(); value = function()
    return value,(time.perf_counter()-start)*1000


def output_hash(output, scope):
    data = bytearray()
    for value in output:
        if scope == "encode":
            data.extend(struct.pack("<Q",len(value)))
            data.extend(struct.pack("<"+"I"*len(value),*value))
        else:
            text = value.encode("utf-8")
            data.extend(struct.pack("<Q",len(text))); data.extend(text)
    return digest(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path); p.add_argument("upstream",type=Path)
    p.add_argument("fixture",type=Path); p.add_argument("profile",choices=["tinystories","gpt2","qwen3","bert"])
    p.add_argument("output",type=Path); a = p.parse_args()
    if a.output.exists(): raise ValueError("output exists")
    if a.fixture.read_bytes() != (ROOT/"tests/fixtures/tokenizers-v1.json").read_bytes(): raise ValueError("unpinned fixture")
    for package,version in VERSIONS.items():
        if importlib.metadata.version(package) != version: raise ValueError(f"unpinned {package}")
    if subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM or subprocess.check_output(["git","-C",str(a.upstream),"status","--porcelain"],text=True).strip():
        raise ValueError("unpinned/dirty upstream")
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
    sys.dont_write_bytecode=True; sys.path.insert(0,str(a.upstream.resolve()))
    from tokenization.bpe_tokenizer import BPETokenizer
    from tokenization.hf_tokenizer import HFTokenizer
    fixture=json.loads(a.fixture.read_bytes()); record=fixture["profiles"][a.profile]; cases=record["cases"]
    if len(cases)!=89: raise ValueError("unexpected workload size")
    lock=json.loads((ROOT/"tools/reference/tokenizer-sources.lock.json").read_bytes())["sources"][a.profile]
    def load():
        # Authentication is inside load timing, just as in the native loader.
        payloads={}
        for name,wanted in lock["files"].items():
            path=a.source/a.profile/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size!=wanted["bytes"]: raise ValueError("invalid source file")
            data=path.read_bytes()
            if digest(data)!=wanted["sha256"]: raise ValueError("source hash mismatch")
            payloads[name]=data
        if a.profile=="tinystories":
            vocab=data_pickle(payloads["tinystories/bpe_vocab_8192.pickle"])
            merges=data_pickle(payloads["tinystories/bpe_merges_8192.pickle"])
            validate_bpe(vocab,merges)
            return BPETokenizer(vocab,merges,{"eos_token":"<|endoftext|>","pad_token":"<|pad|>"})
        tokenizer=HFTokenizer(str((a.source/a.profile).resolve()),local_files_only=True,trust_remote_code=False,use_fast=True,**SPECIAL[a.profile])
        if not tokenizer.hf_tokenizer.is_fast: raise ValueError("slow fallback")
        return tokenizer
    tokenizer,first_load=timed(load); del tokenizer
    loads=[]
    for trial in range(35):
        tokenizer,ms=timed(load)
        if trial>=5: loads.append(ms)
        del tokenizer
    load_host=host(); tokenizer=load()
    texts=[c["text"] for c in cases]; ids=[c["ids"] for c in cases]
    rows=[]
    for scope in ("encode","decode","decode_skip_special"):
        def call():
            if scope=="encode": return [tokenizer.encode(text) for text in texts]
            return [tokenizer.decode(value,scope=="decode_skip_special") for value in ids]
        expected=ids if scope=="encode" else [c["decoded_skip_special" if scope=="decode_skip_special" else "decoded"] for c in cases]
        trials=[]; first_ms=None; first_hash=None
        for trial in range(36):
            output,ms=timed(call)
            if output!=expected: raise ValueError("tokenizer parity mismatch")
            hashed=output_hash(output,scope)
            if trial==0: first_ms=ms; first_hash=hashed
            elif hashed!=first_hash: raise ValueError("nonrepeatable output")
            if trial>=6: trials.append(ms)
            del output
        rows.append({"scope":scope,"first_ms":first_ms,"trials_ms":trials,"output_sha256":first_hash,"host":host()})
    result={"schema_version":1,"runtime":"pinned-python-tokenizers","profile":a.profile,
        "fixture_sha256":digest(a.fixture.read_bytes()),"manifest_sha256":record["manifest_sha256"],"upstream_commit":UPSTREAM,"reference_versions":VERSIONS,
        "source_sha256":{str(path.relative_to(ROOT)):digest(path.read_bytes()) for path in (Path(__file__).resolve(),ROOT/"tools/reference/convert_tokenizers.py",ROOT/"tools/reference/tokenizer-sources.lock.json")},
        "warmups":5,"trials":30,"strings_per_trial":len(cases),"input_utf8_bytes":sum(len(t.encode("utf-8")) for t in texts),"token_positions":sum(map(len,ids)),
        "first_load_ms":first_load,"load_trials_ms":loads,"load_host":load_host,"measurements":rows,"host_final":host(),
        "load_scope":"authenticated original source assets plus pinned upstream wrapper construction; safe data-opcode TinyStories loading; destruction excluded. Different source format from native canonical JSON, not a format-controlled parser comparison",
        "execution_scope":"89 sequential scalar calls, outputs retained; validation, hashing and destruction excluded; encode specials enabled, decode modes separate",
        "cold_disk_cache":False,"device":"cpu"}
    with a.output.open("x") as file: json.dump(result,file,indent=2,allow_nan=False); file.write("\n")


if __name__=="__main__":main()
