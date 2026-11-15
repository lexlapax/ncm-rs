"""Diagnose same-draw TinyStories level-one sampling against a float64 CDF."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--default-logits',type=Path,required=True)
    parser.add_argument('--strict-logits',type=Path,required=True)
    parser.add_argument('--default-samples',type=Path,required=True)
    parser.add_argument('--strict-samples',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    old=load_file(args.default_logits)['logits'].numpy().astype(np.float64)
    new=load_file(args.strict_logits)['logits'].numpy().astype(np.float64)
    if old.shape!=(128,8,4096) or new.shape!=old.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('invalid level-one logits')
    old=old.reshape(-1,4096);new=new.reshape(-1,4096)
    a=json.loads(args.default_samples.read_bytes());b=json.loads(args.strict_samples.read_bytes())
    if a['seed']!=args.seed or b['seed']!=args.seed or a['batch']!=128 or b['batch']!=128:
        raise ValueError('sample seed or batch differs')
    if len(a['codes'])<128 or len(b['codes'])<128:
        raise ValueError('missing first batch')
    ca=np.array([r[1:9] for r in a['codes'][:128]],dtype=np.int64).reshape(-1)
    cb=np.array([r[1:9] for r in b['codes'][:128]],dtype=np.int64).reshape(-1)
    if not np.array_equal([r[0] for r in a['codes'][:128]],[r[0] for r in b['codes'][:128]]):
        raise ValueError('initial code choices differ')
    mask=(1<<64)-1;state=args.seed & mask
    def next64():
        nonlocal state
        state=(state+0x9e3779b97f4a7c15)&mask
        z=state;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z=((z^(z>>27))*0x94d049bb133111eb)&mask
        return z^(z>>31)
    for _ in range(128):next64()  # one initial code per row
    uniform=np.array([next64()>>32 for _ in range(128*8)],dtype=np.float64)/4294967296
    def evaluate(logits):
        p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        return p,(np.cumsum(p,axis=1)>uniform[:,None]).argmax(axis=1)
    pa,ia=evaluate(old);pb,ib=evaluate(new)
    tv=.5*np.abs(pa-pb).sum(axis=1)
    result={'rows':len(uniform),'logit_max_abs':float(np.max(np.abs(old-new))),
            'distribution_tv_max':float(tv.max()),'distribution_tv_mean':float(tv.mean()),
            'same_draw_actual_differences':int(np.count_nonzero(ca!=cb)),
            'same_draw_fp64_cdf_differences':int(np.count_nonzero(ia!=ib)),
            'default_sampler_vs_fp64_cdf':int(np.count_nonzero(ca!=ia)),
            'strict_sampler_vs_fp64_cdf':int(np.count_nonzero(cb!=ib)),
            'inputs_sha256':{'default_logits':sha(args.default_logits),'strict_logits':sha(args.strict_logits),
                             'default_samples':sha(args.default_samples),'strict_samples':sha(args.strict_samples)},
            'scope':'diagnostic same-draw coupling; not a numerical parity or statistical-quality gate'}
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':main()
