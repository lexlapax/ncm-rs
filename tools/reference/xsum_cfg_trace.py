"""Capture pinned XSum guided top-k-1 generator traces on CUDA."""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from xsum_artifact import INPUTS, UPSTREAM, verify_input, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cfg-scale', type=float, required=True)
    parser.add_argument('--prefix-json', type=Path, help='explicit padded input rows')
    parser.add_argument('--decisions-only', action='store_true')
    args = parser.parse_args()
    if not 0.0 < args.cfg_scale <= 100.0:
        raise ValueError('positive finite CFG scale required')
    if args.output.exists() or args.output.is_symlink() or not torch.cuda.is_available():
        raise ValueError('new output and CUDA required')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    if cfg.task != 'xsum':
        raise ValueError('incorrect task')
    evaluator = args.upstream / 'scripts/eval_shard.py'
    evaluator_hash = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'
    if digest_file(evaluator) != evaluator_hash:
        raise ValueError('untrusted evaluator source')
    tree = ast.parse(evaluator.read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('expected one decoder function')
    decoder_code = compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(evaluator), 'exec')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    traces = {}
    def save(name, value):
        if args.decisions_only and name not in ('generator.prefix_ids', 'generator.selected_codes',
                                                'generator.penalized_ids'):
            return
        traces[name] = value.detach().cpu().contiguous().clone()
    levels = list(codec.vq.levels)
    rows = None
    if args.prefix_json:
        rows = json.loads(args.prefix_json.read_bytes())
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 32
                or any(not isinstance(row, list) or len(row) != 512
                       or any(type(v) is not int or not 0 <= v < 50304 for v in row)
                       or all(v == 50257 for v in row) for row in rows)):
            raise ValueError('invalid padded prefixes')
    batch = len(rows) if rows is not None else 2
    selected = []
    upstream_sample = model._sample_level_raw
    if rows is not None:
        prefix = torch.tensor(rows, dtype=torch.long, device='cuda')
    else:
        prefix = torch.full((batch, 512), 50257, dtype=torch.long, device='cuda')
        for row, length in enumerate((24, 497)):
            prefix[row, :length] = (torch.arange(length, device='cuda') * (29 + row * 2) + 3) % 50256
    calls = [0]
    def traced_sample(logits, temperature, top_k, top_p):
        level = calls[0]
        save(f'generator.level_{level:02d}.logits', logits)
        if tuple(logits.shape) != (batch, levels[level], 4096):
            raise ValueError('unexpected generator head shape')
        calls[0] += 1
        codes = upstream_sample(logits, temperature, top_k, top_p)
        selected.append(codes)
        return codes
    model._sample_level_raw = traced_sample
    scope = {'torch': torch, 'freq': 2.0, 'mu': 1.0, 'ADJ': [5.0, 4.0, 0.5],
             'ALLOW': generator_state['inference']['token_unigram'].cuda().double()}
    exec(decoder_code, scope)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        save('generator.prefix_ids', prefix)
        torch.manual_seed(42)
        generated = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                                   use_kv_cache=False, cfg_scale=args.cfg_scale)
        if calls[0] != len(levels) or not torch.equal(generated, torch.cat(selected, dim=1)):
            raise ValueError('generator codes changed')
        save('generator.selected_codes', generated)
        z_hat = torch.zeros(batch, levels[-1], codec.embed_dim, device='cuda')
        for level, codes in enumerate(selected):
            z_hat, _ = codec.vq.get_next_autoregressive_input(level, z_hat, codec.vq.embeddings[level](codes))
            save(f'reconstruction.level_{level:02d}', z_hat)
        final = codec.decoder(z_hat)
        save('generator.final_logits', final)
        save('generator.greedy_ids', final.argmax(-1))
        save('generator.penalized_ids', scope['pick_tokens'](final))
        torch.cuda.synchronize()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if digest_file(evaluator) != evaluator_hash:
        raise ValueError('evaluator source changed')
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(traces, str(args.output / 'trace.safetensors'))
    report = {'upstream': rev, 'sources': {k: v[1] for k, v in INPUTS.items()},
              'torch': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
              'tf32': False, 'sdpa': 'math', 'seed': 42, 'batch': batch,
              'prefix_lengths': ([(sum(v != 50257 for v in row)) for row in rows]
                                 if rows is not None else [24, 497]),
              'prefix_sha256': digest_file(args.prefix_json) if args.prefix_json else None,
              'decisions_only': args.decisions_only,
              'mode': 'upstream temperature 1, top-k 1, top-p 0 guided selection',
              'cfg_scale': args.cfg_scale,
              'trace_sha256': digest_file(args.output / 'trace.safetensors'), 'tensors': sorted(traces)}
    with (args.output / 'report.json').open('x') as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write('\n')


if __name__ == '__main__':
    main()
