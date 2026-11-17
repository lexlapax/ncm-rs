"""Bounded independently generated mathematical erf approximation screen."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import mlx.core as mx
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path);parser.add_argument('--sha',required=True);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--tensor',required=True,type=Path);args=parser.parse_args();p=args.input
if args.output.exists() or args.tensor.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=args.sha:
    raise ValueError('output exists or input digest differs')
d=mx.load(str(p)); step=1/1024
if set(d) != {'x', 'cuda'} or any(d[k].shape != (1, 518, 3072) or d[k].dtype != mx.float32 for k in d):
    raise ValueError('unexpected input shape or dtype')
if not all(np.isfinite(np.asarray(d[k])).all() for k in d):
    raise ValueError('nonfinite input')
xs=np.arange(8193,dtype=np.float64)*step
values=np.array([math.erf(float(x)) for x in xs],dtype=np.float64)
hi=values.astype(np.float32);lo=(values-hi.astype(np.float64)).astype(np.float32)
derivative=(2/math.sqrt(math.pi)*np.exp(-xs*xs)).astype(np.float32)
table=mx.array(np.stack([hi,lo,derivative],axis=1))
src=r'''
uint idx=thread_position_in_grid.x; float v=x[idx];
float z=v*0.7071067811865475244f; float y=metal::abs(z);
float e;
if (y>=8.0f) { e=1.0f; }
else {
 uint i=min(uint(y*1024.0f),8191u); float base=float(i)*(1.0f/1024.0f);
 float delta=y-base; float h=table[i*3]; float l=table[i*3+1]; float slope=table[i*3+2];
 float correction=fma(-base,delta,1.0f); float low=fma(slope*delta,correction,l); e=h+low;
}
e=metal::copysign(e,z); out[idx]=(v*0.5f)*(1.0f+e);
'''
k=mx.fast.metal_kernel(name='ncm_erf_math_table_screen',input_names=['x','table'],output_names=['out'],source=src,ensure_row_contiguous=True,compile_options={'math_mode':'safe'})
(y,)=k(inputs=[d['x'],table],grid=(1591296,1,1),threadgroup=(256,1,1),output_shapes=[(1,518,3072)],output_dtypes=[mx.float32]);a,b=np.asarray(y),np.asarray(d['cuda']);err=np.abs(a-b)
if hashlib.sha256(p.read_bytes()).hexdigest()!=args.sha:
    raise ValueError('input changed')
mx.save_safetensors(str(args.tensor),{'gelu':y})
args.output.write_text(json.dumps({'scope':'one captured row, mathematical approximation; not a gate','input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exact':int((err==0).sum()),'total':int(err.size),'max_abs':float(err.max()),'mean_abs':float(err.mean())},indent=2)+'\n')
