"""Pinned-Python TinyStories allocator observation; not a timing gate."""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from tinystories_artifact import INPUTS, UPSTREAM, verify_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--clear-cuda-cache', action='store_true')
    parser.add_argument('--diagnose-memory', action='store_true')
    parser.add_argument('--collect-garbage', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.batch <= 128 or not torch.cuda.is_available():
        raise ValueError('batch must be 1..128 and CUDA must be available')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, digest) in INPUTS.items():
        verify_input(args.source / name, size, digest)
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codec_data = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    model_data = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_data['model'], strict=True)
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(model_data['model'], strict=True)
    model.eval()
    unigram = model_data['inference']['level0_unigram']
    prefix = torch.zeros((args.batch, 1), dtype=torch.long, device='cuda')
    samples, cleanup = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for trial in range(2):
            if args.diagnose_memory:
                print(json.dumps({'trial': trial, 'allocated': torch.cuda.memory_allocated(),
                                  'reserved': torch.cuda.memory_reserved()}), file=sys.stderr, flush=True)
            torch.manual_seed(42)
            if args.clear_cuda_cache:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            initial = torch.multinomial(unigram, args.batch, replacement=True).reshape(args.batch, 1).cuda()
            codes = model.generate(initial, prefix, temperature=1.0, top_k=0, top_p=1.0,
                                   use_kv_cache=False, cfg_scale=0.0)
            zhats, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            ids = codec.decoder(levels[-1]).argmax(-1)
            codes_cpu, ids_cpu = codes.cpu(), ids.cpu()
            elapsed = time.perf_counter() - start
            if codes_cpu.shape != (args.batch, 641) or ids_cpu.shape != (args.batch, 128):
                raise ValueError('wrong output shape')
            if trial >= 1:
                samples.append(elapsed)
            # Do not retain a previous batch's GPU outputs into the next timed
            # request; the Rust benchmark also drops each request's outputs.
            del initial, codes, zhats, levels, ids, codes_cpu, ids_cpu
            if args.collect_garbage:
                cleanup_start = time.perf_counter()
                gc.collect()
                if trial >= 1:
                    cleanup.append(time.perf_counter() - cleanup_start)
    report = {'batch': args.batch, 'warmups': 1, 'trials_seconds': samples,
              'scope': 'generation plus code and final-ID host readback; model loading excluded',
              'rng': 'PyTorch multinomial, seed 42', 'status': 'exploratory',
              'clear_cuda_cache_between_trials': args.clear_cuda_cache, 'collect_garbage_between_trials': args.collect_garbage,
              'cleanup_seconds': cleanup,
              'torch': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
              'current_torch_allocated_bytes': torch.cuda.memory_allocated(),
              'current_torch_reserved_bytes': torch.cuda.memory_reserved(),
              'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(),
              'tf32': False, 'sdpa': 'PyTorch default'}
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
