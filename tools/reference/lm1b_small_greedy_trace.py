"""Capture pinned upstream LM1B-small greedy free-generation traces on CUDA."""
import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, verify_input
from lm1b_inventory import CHECKPOINTS
from lm1b_export_preparation import CONFIGS

INPUTS = {'tokenizer.pt': CHECKPOINTS['tokenizer.pt'], 'small/generator.pt': CHECKPOINTS['small/generator.pt'], 'small/config.yaml': CONFIGS['small']}

PREFIX_SHA = '5c961e193f0ea101afc2282f673eb3fde02ae6c752e8dd6dde069c3b25a8f90d'


def fixed_codes(data):
    if (data.get('size'), data.get('seed'), data.get('count'), data.get('batch_size')) != ('small', 11, 32, 8):
        raise ValueError('invalid LM1B development capture metadata')
    rows = data.get('codes')
    if (type(rows) is not list or len(rows) != 32 or any(type(row) is not list or len(row) != 641 or
            any(type(x) is not int or not 0 <= x < 8192 for x in row) for row in rows)):
        raise ValueError('invalid LM1B development code rows')
    return rows[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefixes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.output.is_symlink() or out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_VISIBLE_DEVICES',
                                      'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('unsupported environment override')
    if torch.__version__.split('+')[0] != '2.15.0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('pinned Torch and exactly one CUDA device required')
    upstream = args.upstream.resolve()
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != UPSTREAM or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if args.prefixes.is_symlink() or digest_file(args.prefixes) != PREFIX_SHA:
        raise ValueError('prefix reference differs')
    code_rows = fixed_codes(json.loads(args.prefixes.read_bytes()))
    evaluator = upstream / 'scripts/eval_shard.py'
    evaluator_sha = 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a'
    if digest_file(evaluator) != evaluator_sha:
        raise ValueError('unexpected evaluator source')
    tree = ast.parse(evaluator.read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'pick_tokens']
    if len(funcs) != 1:
        raise ValueError('unexpected decoder function')
    decoder = compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(evaluator), 'exec')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_inventory.py').resolve(),
               Path(__file__).with_name('lm1b_export_preparation.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), args.prefixes.resolve(), evaluator.resolve()]
    watched += sorted((upstream / 'models').glob('*.py'))
    before = {str(p): digest_file(p) for p in watched}
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'],
                                  text=True).strip().splitlines()
    if len(gpu) != 1:
        raise ValueError('GPU identity is ambiguous')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'small/config.yaml').read_bytes()))
    if cfg.task != 'lm1b-small':
        raise ValueError('unexpected task')
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'small/generator.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    scope = {'torch': torch, 'freq': 11.5, 'mu': 2.0, 'ADJ': [10.0, 4.0, 0.5],
             'ALLOW': generator_state['inference']['token_unigram'].double().cuda()}
    exec(decoder, scope)
    traces = {}
    def save(name, value):
        if name in traces or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError(f'duplicate or nonfinite trace: {name}')
        traces[name] = value.detach().cpu().contiguous().clone()
    levels = list(codec.vq.levels)
    if levels != [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]:
        raise ValueError('unexpected level geometry')
    all_codes = torch.tensor(code_rows, dtype=torch.int64, device='cuda')
    forced = list(all_codes.split(levels, dim=1))
    calls = [1]
    def record_greedy(logits, temperature, top_k, top_p):
        level = calls[0]
        if level >= len(levels) or tuple(logits.shape) != (2, levels[level], 8192):
            raise ValueError('unexpected head shape')
        save(f'generator.level_{level:02d}.logits', logits)
        calls[0] += 1
        if temperature != 0:
            raise ValueError("expected greedy temperature")
        return logits.argmax(dim=-1)
    model._sample_level_raw = record_greedy
    handles = []
    for level in range(1, len(levels)):
        def hook(_module, args, output, level=level):
            save(f'generator.level_{level:02d}.post_conv_input', args[0])
            save(f'generator.level_{level:02d}.post_conv_output', output)
        handles.append(model.post_upsample_convs[level].register_forward_hook(hook))
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        prefix = torch.full((2, 1), 101, dtype=torch.int64, device='cuda')
        save('generator.prefix_ids', prefix)
        save('generator.initial_codes', forced[0])
        codes = model.generate(forced[0], prefix, temperature=0.0, top_k=1, top_p=0.0,
                               use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=101)
        if calls[0] != len(levels) or not torch.equal(codes[:, :1], forced[0]):
            raise ValueError('unexpected greedy history')
        save('generator.generated_codes', codes)
        generated = list(codes.split(levels, dim=1))
        z = torch.zeros(2, levels[-1], codec.embed_dim, device='cuda')
        for level, indices in enumerate(generated):
            z, _ = codec.vq.get_next_autoregressive_input(level, z, codec.vq.embeddings[level](indices))
            save(f'reconstruction.level_{level:02d}', z)
        final = codec.decoder(z)
        save('generator.final_logits', final)
        save('generator.greedy_ids', final.argmax(-1))
        save('generator.penalized_ids', scope['pick_tokens'](final))
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if (before != {str(p): digest_file(p) for p in watched} or
            subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != head or
            subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('source changed during capture')
    out.mkdir(parents=True)
    save_file(traces, out / 'trace.safetensors')
    report = {'scope': 'two LM1B-small greedy histories with supplied development level zero; not stochastic quality',
              'upstream': head, 'sources': before, 'checkpoint_sha256': {k: v[1] for k, v in INPUTS.items()},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'gpu_identity_and_driver': gpu[0], 'gpu_name': torch.cuda.get_device_name(0),
              'tf32': False, 'sdpa': 'math', 'code_reference_sha256': PREFIX_SHA,
              'trace_sha256': digest_file(out / 'trace.safetensors'), 'tensors': sorted(traces)}
    json_file(out / 'report.json', report)
    print(json.dumps({'tensors': len(traces), 'trace_sha256': report['trace_sha256'],
                      'report_sha256': digest_file(out / 'report.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
