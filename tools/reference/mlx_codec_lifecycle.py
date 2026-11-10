"""Fresh-process adapted MLX lifecycle; independent of steady-state timings."""
import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_codec import Codec, prepare_device
from mlx_codec_bench import check_baseline, call, data, ids_hash


def emit(phase, start, digest=None):
    print(json.dumps({'schema_version':1,'phase':phase,'internal_ms':(time.perf_counter()-start)*1000,
                      'output_ids_sha256':digest}), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('qualified_baseline',type=Path)
    a=p.parse_args();check_baseline(a.qualified_baseline)
    start=time.perf_counter();prepare_device();mx.synchronize();emit('context',start)
    start=time.perf_counter();model=Codec.load(a.bundle);mx.synchronize();emit('load',start)
    ids,_=data(1,model.levels)
    start=time.perf_counter();result=call(model,'reconstruct',ids,[],1,None)
    if len(result[1])!=411 or result[1][322:]!=ids: raise ValueError('incorrect solved-board output')
    emit('execute',start,ids_hash(result[1]))
    start=time.perf_counter();del result,model;mx.synchronize();emit('drop',start)


if __name__=='__main__':main()
