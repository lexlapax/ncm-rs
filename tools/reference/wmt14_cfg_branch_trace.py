"""Trace conditioned and null heads in the divergent WMT14 CFG batch."""
import argparse
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
from wmt14_inventory import INPUTS

PREFIX_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'
FULL_SHA = '78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256'
START = 2180


def prefixes(data):
    rows = data['prefixes']
    if len(rows) != 3003 or any(type(row) is not list or len(row) != 96 or
                                 any(type(x) is not int or not 0 <= x < 151680 for x in row)
                                 for row in rows[START:START + 4]):
        raise ValueError('invalid WMT14 prefix reference')
    return rows[START:START + 4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefixes', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
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
    if args.full.is_symlink() or digest_file(args.full) != FULL_SHA:
        raise ValueError('full reference differs')
    input_rows = prefixes(json.loads(args.prefixes.read_bytes()))
    full = json.loads(args.full.read_bytes())
    if full['start'] != 0 or full['count'] != 3003 or full['batch_size'] != 4 or full['cfg_scale'] != 1.0:
        raise ValueError('unexpected full capture')
    expected = full['outputs'][START:START + 4]
    watched = [Path(__file__).resolve(), Path(__file__).with_name('wmt14_inventory.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), args.prefixes.resolve(), args.full.resolve()]
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
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    generation = cfg.generation
    if (cfg.task != 'wmt14' or generation.temperature != 1.0 or generation.top_k != 1 or
            generation.top_p != 0.0 or generation.cfg_scale != 0.0 or
            generation.get('frequency_penalty', 0) or generation.get('adjacency_penalty')):
        raise ValueError('unexpected task')
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
        if name in traces or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError(f'duplicate or nonfinite trace: {name}')
        traces[name] = value.detach().cpu().contiguous().clone()
    handles = []
    for level, head_module in enumerate([model.lm_head_l0, *model.lm_heads]):
        def observe_head(_module, inputs, output, level=level):
            save(f'generator.level_{level:02d}.head_input', inputs[0])
            save(f'generator.level_{level:02d}.branch_logits', output)
        handles.append(head_module.register_forward_hook(observe_head))
    levels = list(codec.vq.levels)
    if levels != [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]:
        raise ValueError('unexpected level geometry')
    calls = [0]
    for stage_index, stage_module in enumerate(codec.vq.learned_up_l1):
        def observe_cascade(_module, inputs, output, stage_index=stage_index):
            if calls[0] == 1:
                save(f'diagnostic.cascade_{stage_index:02d}.input', inputs[0].transpose(-1, -2))
                save(f'diagnostic.cascade_{stage_index:02d}.output', output.transpose(-1, -2))
        handles.append(stage_module.register_forward_hook(observe_cascade))
    original_next = codec.vq.get_next_autoregressive_input
    def observe_next(level, z_hat, selected):
        value, following = original_next(level, z_hat, selected)
        if level == 0 and calls[0] == 1:
            save('diagnostic.level_00.generated_cumulative', value)
        return value, following
    codec.vq.get_next_autoregressive_input = observe_next
    for block_index, block in enumerate(model.transformer.blocks):
        def observe_block(_module, inputs, output, block_index=block_index):
            if calls[0] == 5:
                if block_index == 0:
                    save('generator.level_05.stack_input', inputs[0])
                save(f'generator.level_05.block_{block_index:02d}.output', output)
        handles.append(block.register_forward_hook(observe_block))
    original_sample = model._sample_level_raw
    def observe(logits, temperature, top_k, top_p):
        level = calls[0]
        if tuple(logits.shape) != (4, levels[level], 4096):
            raise ValueError('unexpected head shape')
        save(f'generator.level_{level:02d}.logits', logits)
        calls[0] += 1
        return original_sample(logits, temperature, top_k, top_p)
    model._sample_level_raw = observe
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        prefix = torch.tensor(input_rows, dtype=torch.int64, device='cuda')
        save('generator.prefix_ids', prefix)
        codes = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                               use_kv_cache=False, cfg_scale=1.0, cfg_null_eos=151645)
        if calls[0] != len(levels) or tuple(codes.shape) != (4, sum(levels)):
            raise ValueError('unexpected generated history')
        if codes.tolist() != [row['codes'] for row in expected]:
            raise ValueError('same-batch codes differ from full capture')
        save('generator.codes', codes)
        z = torch.zeros(4, levels[-1], codec.embed_dim, device='cuda')
        for level, indices in enumerate(torch.split(codes, levels, dim=1)):
            z, _ = codec.vq.get_next_autoregressive_input(level, z, codec.vq.embeddings[level](indices))
            save(f'reconstruction.level_{level:02d}', z)
        final = codec.decoder(z)
        save('generator.final_logits', final)
        save('generator.greedy_ids', final.argmax(-1))
        if final.argmax(-1).tolist() != [row['raw_ids'] for row in expected]:
            raise ValueError('same-batch IDs differ from full capture')
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
    report = {'scope': 'original four-row CFG=1 batch containing row 2181; diagnostic, not quality or performance',
              'upstream': head, 'sources': before, 'checkpoint_sha256': {k: v[1] for k, v in INPUTS.items()},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'gpu_identity_and_driver': gpu[0], 'gpu_name': torch.cuda.get_device_name(0),
              'tf32': False, 'sdpa': 'math', 'cfg_scale': 1.0, 'cfg_null_eos': 151645, 'prefix_reference_sha256': PREFIX_SHA,
              'trace_sha256': digest_file(out / 'trace.safetensors'), 'tensors': sorted(traces)}
    json_file(out / 'report.json', report)
    print(json.dumps({'tensors': len(traces), 'trace_sha256': report['trace_sha256'],
                      'report_sha256': digest_file(out / 'report.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
