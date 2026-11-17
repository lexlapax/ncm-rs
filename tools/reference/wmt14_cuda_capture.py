"""Capture a bounded contiguous range of pinned upstream WMT14 CUDA outputs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinystories_artifact import ROOT, UPSTREAM, digest_file, json_file, verify_input
from wmt14_inventory import INPUTS

TOKENIZER_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefixes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 3003:
        raise ValueError('invalid source row range')
    if torch.__version__.split('+')[0] != '2.15.0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('pinned Torch and one CUDA device required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_VISIBLE_DEVICES',
                                      'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')):
        raise ValueError('unsupported environment override')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'],
                                  text=True).strip().splitlines()
    if len(gpu) != 1:
        raise ValueError('ambiguous GPU identity')
    output = args.output.resolve()
    if (args.output.is_symlink() or output.exists() or
            not output.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('output must be a new file under reports')
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
    if revision != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if args.prefixes.is_symlink() or digest_file(args.prefixes) != TOKENIZER_SHA:
        raise ValueError('tokenizer reference differs')
    source_files = [Path(__file__).resolve(), Path(__file__).with_name('wmt14_inventory.py').resolve(),
                    Path(__file__).with_name('tinystories_artifact.py').resolve(),
                    upstream / 'scripts/eval_shard.py'] + sorted((upstream / 'models').glob('*.py'))
    before = {str(path): digest_file(path) for path in source_files}
    data = json.loads(args.prefixes.read_bytes())
    rows = data['prefixes'][args.start:args.start + args.count]
    if len(data['prefixes']) != 3003 or len(rows) != args.count or any(len(prefix) != 96 or any(type(x) is not int or x < 0 or x >= 151680 for x in prefix) for prefix in rows):
        raise ValueError('unexpected prefix shape')
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    generation = cfg.generation
    if (cfg.task != 'wmt14' or generation.temperature != 1.0 or generation.top_k != 1 or
            generation.top_p != 0.0 or generation.cfg_scale != 0.0 or
            generation.get('frequency_penalty', 0) or generation.get('adjacency_penalty')):
        raise ValueError('unsupported generation configuration')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    codec_state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    generator_state = torch.load(args.source / 'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(codec_state['model'], strict=True)
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    outputs = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for start in range(0, args.count, 4):
            prefix = torch.tensor(rows[start:start + 4], dtype=torch.long, device='cuda')
            codes = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                                   use_kv_cache=False, cfg_scale=0.0)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            ids = codec.decoder(levels[-1]).argmax(-1)
            for index, (code_values, id_values) in enumerate(zip(codes.tolist(), ids.tolist(), strict=True), args.start + start):
                if (len(code_values) != 313 or len(id_values) != 64 or
                        any(not 0 <= x < 4096 for x in code_values) or any(not 0 <= x < 151680 for x in id_values)):
                    raise ValueError('unexpected output shape or range')
                outputs.append({'row': index, 'codes': code_values, 'raw_ids': id_values})
            print(f'{len(outputs)}/{args.count}', flush=True)
    torch.cuda.synchronize()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if digest_file(args.prefixes) != TOKENIZER_SHA or before != {str(p): digest_file(p) for p in source_files}:
        raise ValueError('probe inputs changed')
    if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != revision or
            subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('upstream checkout changed')
    json_file(output, {'scope': 'upstream CUDA output capture; scoring and performance are separate',
                       'upstream': revision, 'torch': torch.__version__, 'prefix_reference_sha256': TOKENIZER_SHA,
                       'source_sha256': before, 'checkpoint_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
                       'device': torch.cuda.get_device_name(0), 'gpu_identity_and_driver': gpu[0], 'cuda': torch.version.cuda,
                       'tf32': False, 'sdpa': 'math', 'batch_size': 4, 'start': args.start,
                       'count': args.count, 'outputs': outputs})
    print(json.dumps({'rows': len(outputs),
                      'output_sha256': digest_file(output)}, sort_keys=True))


if __name__ == '__main__':
    main()
