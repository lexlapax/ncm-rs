"""Synchronized per-level Sudoku diagnostic; instrumentation changes timing."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from sudoku_artifact import INPUTS, INPUT_SIZES, UPSTREAM, checkpoint

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
p.add_argument('--kernel-output', type=Path)
a=p.parse_args()
if a.output.exists() or not torch.cuda.is_available(): raise ValueError('output must be new and CUDA available')
if subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM: raise ValueError('upstream revision differs')
if subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip(): raise ValueError('upstream is dirty')
sources={}
for name, digest in INPUTS.items():
    path=a.source/name
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=INPUT_SIZES[name]: raise ValueError('invalid source file')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('source digest differs')
    sources[name]=data
cfg=OmegaConf.create(yaml.safe_load(sources['config.yaml']))
rows=json.loads(a.inputs.read_bytes())
if not isinstance(rows,list) or not 1<=len(rows)<=128 or any(not isinstance(row,list) or len(row)!=91 or any(type(x) is not int or not 0<=x<12 for x in row) for row in rows): raise ValueError('invalid inputs')
sys.dont_write_bytecode=True; sys.path.insert(0,str(a.upstream.resolve()))
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
codec=VQVAE(**cfg.tokenizer.vqvae);codec.load_state_dict(checkpoint(sources['tokenizer.pt']),strict=True)
model=NextConceptTransformer(codec,cfg.model);model.load_state_dict(checkpoint(sources['generator.pt']),strict=True)
codec,model=codec.cuda().eval(),model.cuda().eval()
records=[]; state={}; handles=[]
for head in model.lm_heads:
    def collect(_module,_args,output):
        output.detach().cpu()
        now=time.perf_counter();state['levels'].append(now-state['previous']);state['previous']=now
    handles.append(head.register_forward_hook(collect))
with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
    for _ in range(4):
        torch.cuda.synchronize();state={'previous':time.perf_counter(),'levels':[]}
        prefix=torch.tensor(rows,dtype=torch.long,device='cuda')
        codes=model.generate(None,prefix,temperature=0,top_k=0,top_p=1,use_kv_cache=False,cfg_scale=0)
        ids=codec.decode(codes).argmax(-1)
        codes.cpu();ids.cpu();now=time.perf_counter()
        records.append({'levels_seconds':state['levels'],'tail_seconds':now-state['previous']})
for handle in handles:handle.remove()
if a.kernel_output:
    from torch.profiler import profile, ProfilerActivity
    if a.kernel_output.exists(): raise ValueError('kernel output exists')
    prefix=torch.tensor(rows,dtype=torch.long,device='cuda')
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            codes=model.generate(None,prefix,temperature=0,top_k=0,top_p=1,use_kv_cache=False,cfg_scale=0)
            ids=codec.decode(codes).argmax(-1); codes.cpu();ids.cpu();torch.cuda.synchronize()
    kernels={}
    for event in prof.events():
        if 'cuda' not in str(event.device_type).lower(): continue
        entry=kernels.setdefault(event.name,{'count':0,'device_us':0.0})
        entry['count']+=1;entry['device_us']+=event.device_time_total
    if not kernels: raise ValueError('no GPU kernel events captured')
    cpu=sorted(({'name':event.key,'self_cpu_us':event.self_cpu_time_total,'count':event.count} for event in prof.key_averages()),key=lambda x:x['self_cpu_us'],reverse=True)[:30]
    with a.kernel_output.open('x') as f:json.dump({'kernels':kernels,'top_cpu':cpu,'scope':'instrumented single call; device event sums may overlap'},f,indent=2);f.write('\n')
with a.output.open('x') as f:
    json.dump({'batch':len(rows),'runs':records,'scope':'instrumented: each level reads all logits and synchronizes'},f,indent=2);f.write('\n')
