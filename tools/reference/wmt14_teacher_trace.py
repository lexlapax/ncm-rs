"""Capture pinned upstream WMT14 fixed-history generator traces on CUDA."""
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


def prefixes(data):
    rows = data['prefixes']
    if len(rows) != 3003 or any(type(row) is not list or len(row) != 96 or
                                 any(type(x) is not int or not 0 <= x < 151680 for x in row)
                                 for row in rows[:2]):
        raise ValueError('invalid WMT14 prefix reference')
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
    input_rows = prefixes(json.loads(args.prefixes.read_bytes()))
    watched = [Path(__file__).resolve(), Path(__file__).with_name('wmt14_inventory.py').resolve(),
               Path(__file__).with_name('tinystories_artifact.py').resolve(), args.prefixes.resolve()]
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
    if cfg.task != 'wmt14':
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
    levels = list(codec.vq.levels)
    if levels != [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]:
        raise ValueError('unexpected level geometry')
    forced = [((torch.arange(2 * length, device='cuda').reshape(2, length) * 37 + level * 97) % 4096).long()
              for level, length in enumerate(levels)]
    calls = [0]
    def force(logits, temperature, top_k, top_p):
        level = calls[0]
        if tuple(logits.shape) != (2, levels[level], 4096):
            raise ValueError('unexpected head shape')
        save(f'generator.level_{level:02d}.logits', logits)
        calls[0] += 1
        return forced[level]
    model._sample_level_raw = force
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        prefix = torch.tensor(input_rows, dtype=torch.int64, device='cuda')
        save('generator.prefix_ids', prefix)
        codes = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                               use_kv_cache=False, cfg_scale=0.0)
        if calls[0] != len(levels) or not torch.equal(codes, torch.cat(forced, dim=1)):
            raise ValueError('forced history differs')
        save('generator.forced_codes', codes)
        z = torch.zeros(2, levels[-1], codec.embed_dim, device='cuda')
        for level, indices in enumerate(forced):
            z, _ = codec.vq.get_next_autoregressive_input(level, z, codec.vq.embeddings[level](indices))
            save(f'reconstruction.level_{level:02d}', z)
        final = codec.decoder(z)
        save('generator.final_logits', final)
        save('generator.greedy_ids', final.argmax(-1))
        torch.cuda.synchronize()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if (before != {str(p): digest_file(p) for p in watched} or
            subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != head or
            subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('source changed during capture')
    out.mkdir(parents=True)
    save_file(traces, out / 'trace.safetensors')
    report = {'scope': 'two WMT14 prefixes with fixed code histories, not sampling or quality',
              'upstream': head, 'sources': before, 'checkpoint_sha256': {k: v[1] for k, v in INPUTS.items()},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'gpu_identity_and_driver': gpu[0], 'gpu_name': torch.cuda.get_device_name(0),
              'tf32': False, 'sdpa': 'math', 'prefix_reference_sha256': PREFIX_SHA,
              'trace_sha256': digest_file(out / 'trace.safetensors'), 'tensors': sorted(traces)}
    json_file(out / 'report.json', report)
    print(json.dumps({'tensors': len(traces), 'trace_sha256': report['trace_sha256'],
                      'report_sha256': digest_file(out / 'report.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
