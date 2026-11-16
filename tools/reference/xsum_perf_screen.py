"""Exploratory pinned-Python XSum timing, not a quality or acceptance gate."""
import argparse
import ast
import contextlib
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from xsum_artifact import INPUTS, UPSTREAM, digest_file, verify_input


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--no-stage-sync', action='store_true')
    ap.add_argument('--trials', type=int, choices=(5, 30), default=5)
    ap.add_argument('--collect-garbage', action='store_true')
    ap.add_argument('--cfg-scale', type=float, default=0.0)
    args = ap.parse_args()
    if not 1 <= args.batch <= 128 or not 0.0 <= args.cfg_scale <= 100.0 or not torch.cuda.is_available():
        raise ValueError('batch 1..128 and CUDA required')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    rows = json.loads(args.inputs.read_bytes())[:args.batch]
    if len(rows) != args.batch or any(len(row) != 512 for row in rows):
        raise ValueError('need 512-token input rows')
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    evaluator = args.upstream / 'scripts/eval_shard.py'
    if digest_file(evaluator) != 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a':
        raise ValueError('unexpected evaluator source')
    tree = ast.parse(evaluator.read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('expected one decoder function')
    decoder = compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(evaluator), 'exec')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    with contextlib.redirect_stdout(sys.stderr):
        codec = VQVAE(**cfg.tokenizer.vqvae)
        codec.load_state_dict(codec_state['model'], strict=True)
        codec = codec.cuda().eval()
        model = NextConceptTransformer(codec, cfg.model).cuda()
        model.load_state_dict(generator_state['model'], strict=True)
        model.eval()
    scope = {'torch': torch, 'freq': 2.0, 'mu': 1.0, 'ADJ': [5.0, 4.0, 0.5],
             'ALLOW': generator_state['inference']['token_unigram'].cuda().double()}
    exec(decoder, scope)
    del codec_state, generator_state
    prefix = torch.tensor(rows, device='cuda', dtype=torch.long)
    samples, decoder_samples, cleanup_samples, signature = [], [], [], None
    warmups = 5 if args.trials == 30 else 2
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        torch.manual_seed(42)
        for trial in range(warmups + args.trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            codes = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                                   use_kv_cache=False, cfg_scale=args.cfg_scale)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            logits = codec.decoder(levels[-1])
            if not args.no_stage_sync:
                torch.cuda.synchronize()
                decoder_elapsed = time.perf_counter() - start
            ids = scope['pick_tokens'](logits)
            current = (codes.tolist(), ids.tolist())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if signature is not None and signature != current:
                raise ValueError('generation changed between trials')
            signature = current
            del codes, levels, ids, logits, _
            if args.collect_garbage:
                cleanup_start = time.perf_counter()
                gc.collect()
                torch.cuda.synchronize()
                cleanup_elapsed = time.perf_counter() - cleanup_start
            if trial >= warmups:
                samples.append(elapsed)
                if not args.no_stage_sync:
                    decoder_samples.append(decoder_elapsed)
                if args.collect_garbage:
                    cleanup_samples.append(cleanup_elapsed)
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if digest_file(evaluator) != 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a':
        raise ValueError('evaluator source changed')
    print(json.dumps({'batch': args.batch, 'cfg_scale': args.cfg_scale, 'warmups': warmups, 'trials_seconds': samples,
                      'decoder_seconds': decoder_samples,
                      'cleanup_seconds': cleanup_samples,
                      'scope': 'generation, FP64 final selection and code/ID host readback; load excluded; optional decoder synchronization',
                      'status': 'exploratory', 'backend': 'pinned Python CUDA',
                      'codes': signature[0], 'ids': signature[1],
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved()}))


if __name__ == '__main__':
    main()
