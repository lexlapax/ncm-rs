"""Paired Python-MLX retained-root memory observation; not acceptance."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'tools/reference'))
import mlx.core as mx
import numpy as np
from mlx_codec_bench import Codec, call, check_baseline, data, ids_hash, prepare_device

def snap():
    return {'active':mx.get_active_memory(),'cache':mx.get_cache_memory(),'peak':mx.get_peak_memory()}

def main():
    if len(sys.argv)!=3 or Path(sys.argv[2]).exists():
        raise ValueError('usage: probe BUNDLE NEW_REPORT')
    bundle=Path(sys.argv[1]); output=Path(sys.argv[2])
    baseline=check_baseline(ROOT/'reports/phase2-codec/python-mlx-v1')
    prepare_device()
    one=mx.array([1.],dtype=mx.float32);mx.eval(one)
    codec=Codec.load(bundle);rows=[]
    for batch in (1,2,4):
        tokens,codes=data(batch,codec.levels)
        encoded,snapshots=codec.encode(mx.array(tokens,dtype=mx.uint32).reshape(batch,89))
        latent=snapshots[-1];mx.eval(latent);del encoded,snapshots
        latent_sha256=hashlib.sha256(np.asarray(latent,dtype='<f4').tobytes()).hexdigest()
        for scope in ('encode','decode','reconstruct','resident_decoder_block'):
            mx.reset_peak_memory()
            first=call(codec,scope,tokens,codes,batch,latent)
            floats,ids,logits=first;output_hash=ids_hash(ids)
            if logits is not None: logits.reshape(-1).tolist()
            del first,floats,ids,logits
            for _ in range(35):
                result=call(codec,scope,tokens,codes,batch,latent)
                if ids_hash(result[1])!=output_hash:raise ValueError('nonrepeatable IDs')
                del result
            immediate=snap();barrier=one+one;mx.eval(barrier);del barrier;mx.synchronize();synchronized=snap()
            time.sleep(.01);after_10ms=snap();time.sleep(.09);after_100ms=snap()
            rows.append({'batch':batch,'scope':scope,'input_ids_sha256':ids_hash(tokens),'supplied_codes_sha256':ids_hash(codes),'output_ids_sha256':output_hash,'latent_sha256':latent_sha256,'immediate':immediate,'synchronized':synchronized,'after_10ms':after_10ms,'after_100ms':after_100ms})
        del latent
    del codec,one
    gc.collect();barrier=mx.array([1.],dtype=mx.float32);mx.eval(barrier+barrier);del barrier;mx.synchronize();time.sleep(.1)
    report={'schema_version':1,'runtime':'python-mlx-adapted','scope':'finite retained-root memory diagnostic; not acceptance','manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),'baseline_qualification_sha256':baseline,'warmups_and_trials':35,'rows':rows,'after_model_drop':snap()}
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
