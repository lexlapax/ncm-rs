"""Authenticated fixed Sudoku assets for the offline Python MLX codec baseline."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "c70274273f489abc73832b2b98584a8483b6b203"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensors(data):
    if len(data)<8: raise ValueError("short safetensors header")
    size=int.from_bytes(data[:8],"little")
    if size>len(data)-8: raise ValueError("safetensors header out of bounds")
    header=json.loads(data[8:8+size]); arrays={}
    ranges=[]
    for name,entry in header.items():
        if name=="__metadata__":continue
        dtype={"F32":"<f4","I64":"<i8"}.get(entry["dtype"])
        if dtype is None:raise ValueError("unsupported reference tensor dtype")
        shape=entry["shape"];begin,end=entry["data_offsets"]
        if any(type(d) is not int or d<=0 for d in shape) or len(shape)>8 or not 0<=begin<=end<=len(data)-8-size:
            raise ValueError("invalid reference tensor layout")
        if math.prod(shape)*np.dtype(dtype).itemsize!=end-begin:raise ValueError("tensor byte count differs")
        raw=memoryview(data)[8+size+begin:8+size+end]
        arrays[name]=np.frombuffer(raw,dtype=dtype).reshape(shape)
        ranges.append((begin,end))
    offset=0
    for begin,end in sorted(ranges):
        if begin!=offset:raise ValueError("gapped/overlapping tensor payload")
        offset=end
    if offset!=len(data)-8-size:raise ValueError("trailing tensor bytes")
    return arrays


def load_bundle(root):
    anchor=json.loads((ROOT/"docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json").read_bytes())
    if root.is_symlink() or not root.is_dir():raise ValueError("invalid artifact root")
    manifest_path=root/"manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size>4*1024**2:raise ValueError("invalid manifest")
    manifest=json.loads(manifest_path.read_bytes())
    if manifest!=anchor:raise ValueError("artifact differs from pinned Sudoku inventory")
    payloads={}
    for name,record in manifest["files"].items():
        path=root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=record["bytes"]:raise ValueError("invalid artifact file")
        data=path.read_bytes()
        if digest(data)!=record["sha256"]:raise ValueError("artifact hash mismatch")
        if name.endswith(".json"):json.loads(data)
        payloads[name]=data
    runtime=None
    for filename,records in manifest["tensors"].items():
        arrays=tensors(payloads[filename])
        if set(arrays)!={r["name"] for r in records}:raise ValueError("tensor inventory differs")
        for record in records:
            array=arrays[record["name"]]
            dtype={"F32":np.dtype("<f4"),"I64":np.dtype("<i8")}[record["dtype"]]
            if list(array.shape)!=record["shape"] or array.dtype!=dtype or array.nbytes!=record["bytes"] or digest(memoryview(array))!=record["sha256"]:
                raise ValueError("tensor metadata/hash differs")
        if filename=="codec.safetensors":runtime=arrays
    if runtime is None or len(runtime)!=58 or any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in runtime.values()):
        raise ValueError("invalid runtime weights")
    config=json.loads(payloads["config.json"])["tokenizer"]["vqvae"]
    # Authenticating the entire config limits this adapter to the one pinned profile.
    return config,runtime


def save_tensors(path, arrays):
    """Deterministic simple safetensors export; no pickle or new package dependency."""
    header={};payload=[];offset=0
    for name,array in sorted(arrays.items()):
        array=np.asarray(array)
        if array.dtype==np.float32: dtype="F32";array=array.astype("<f4",copy=False)
        elif array.dtype.kind in "iu":dtype="I64";array=array.astype("<i8")
        else:raise ValueError("unsupported capture dtype")
        data=array.tobytes(order="C")
        header[name]={"dtype":dtype,"shape":list(array.shape),"data_offsets":[offset,offset+len(data)]}
        payload.append(data);offset+=len(data)
    encoded=json.dumps(header,separators=(",",":")).encode();encoded+=b" "*((-len(encoded))%8)
    with path.open("xb") as file:
        file.write(len(encoded).to_bytes(8,"little"));file.write(encoded)
        for data in payload:file.write(data)
