"""Capture CPU and CUDA decisions from the exact pinned XSum evaluator function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import torch

UPSTREAM = 'c70274273f489abc73832b2b98584a8483b6b203'
EVAL_SHA256 = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    source = (args.upstream / 'scripts/eval_shard.py').read_bytes()
    if revision != UPSTREAM or hashlib.sha256(source).hexdigest() != EVAL_SHA256:
        raise ValueError('upstream source differs from the pinned evaluator')
    if not torch.cuda.is_available():
        raise ValueError('CUDA oracle is required')
    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('expected one upstream decoder function')
    module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    code = compile(module, str(args.upstream / 'scripts/eval_shard.py'), 'exec')
    rng = torch.Generator(device='cpu').manual_seed(476)
    cases = []
    for name, batch, length, vocab in [('varied', 3, 8, 19), ('ties', 2, 6, 7), ('repeated', 2, 7, 5)]:
        logits = torch.randn((batch, length, vocab), generator=rng, dtype=torch.float32)
        if name == 'ties':
            logits.zero_()
        if name == 'repeated':
            logits.zero_()
            logits[:, :, 0] = 32.0
            logits[:, -1, 1] = 31.0
        if name == 'ties':
            allowance = torch.full((vocab,), 1.0 / vocab, dtype=torch.float32)
        else:
            allowance = torch.rand((vocab,), generator=rng, dtype=torch.float32)
            allowance = allowance / allowance.sum()
        decisions = {}
        for device in ('cpu', 'cuda'):
            scope = {'torch': torch, 'freq': 2.0, 'mu': 1.0, 'ADJ': [5.0, 4.0, 0.5],
                     'ALLOW': allowance.to(device).double()}
            exec(code, scope)
            decisions[device] = scope['pick_tokens'](logits.to(device)).cpu().tolist()
        if decisions['cpu'] != decisions['cuda']:
            raise ValueError(f'CPU/CUDA oracle decisions differ for {name}')
        cases.append({'name': name, 'batch': batch, 'time': length, 'vocab': vocab,
                      'logits': logits.reshape(-1).tolist(), 'allowance': allowance.tolist(),
                      'expected': [v for row in decisions['cuda'] for v in row]})
    if hashlib.sha256((args.upstream / 'scripts/eval_shard.py').read_bytes()).hexdigest() != EVAL_SHA256:
        raise ValueError('upstream source changed during capture')
    with args.output.open('x') as file:
        json.dump({'schema_version': 1, 'upstream_commit': UPSTREAM, 'evaluator_sha256': EVAL_SHA256,
                   'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(),
                   'cases': cases}, file, sort_keys=True, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
