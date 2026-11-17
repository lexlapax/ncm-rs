"""Measure pinned upstream WMT14 CUDA generation; scoring is separate."""
import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
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
    parser.add_argument('--batch', type=int, choices=[1, 4, 16], required=True)
    parser.add_argument('--cfg', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
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
    rows = data['prefixes'][:args.batch]
    if len(data['prefixes']) != 3003 or len(rows) != args.batch or any(len(prefix) != 96 or any(type(x) is not int or x < 0 or x >= 151680 for x in prefix) for prefix in rows):
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
    del codec_state, generator_state
    samples, device_samples, cleanup_samples, signature = [], [], [], None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for trial in range(35):
            torch.cuda.synchronize()
            device_start = torch.cuda.Event(enable_timing=True)
            device_end = torch.cuda.Event(enable_timing=True)
            device_start.record()
            started = time.perf_counter()
            prefix = torch.tensor(rows, dtype=torch.long, device='cuda')
            codes = model.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                                   use_kv_cache=False, cfg_scale=float(args.cfg), cfg_null_eos=151645)
            _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
            ids = codec.decoder(levels[-1]).argmax(-1)
            device_end.record()
            codes_host, ids_host = codes.cpu(), ids.cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            current = {'codes': codes_host.tolist(), 'ids': ids_host.tolist()}
            if signature is not None and current != signature:
                raise ValueError('output changed between trials')
            signature = current
            del prefix, codes, levels, ids, codes_host, ids_host, _
            cleanup_start = time.perf_counter()
            gc.collect()
            torch.cuda.synchronize()
            cleanup = time.perf_counter() - cleanup_start
            if trial >= 5:
                samples.append(elapsed)
                device_samples.append(device_start.elapsed_time(device_end) / 1000.0)
                cleanup_samples.append(cleanup)
    torch.cuda.synchronize()
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    if digest_file(args.prefixes) != TOKENIZER_SHA or before != {str(p): digest_file(p) for p in source_files}:
        raise ValueError('probe inputs changed')
    if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != revision or
            subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('upstream checkout changed')
    json_file(output, {'scope': 'generation and codes/IDs host readback; model load and cleanup excluded',
                       'upstream': revision, 'torch': torch.__version__, 'prefix_reference_sha256': TOKENIZER_SHA,
                       'source_sha256': before, 'checkpoint_sha256': {name: sha for name, (_, sha) in INPUTS.items()},
                       'device': torch.cuda.get_device_name(0), 'gpu_identity_and_driver': gpu[0], 'cuda': torch.version.cuda,
                       'tf32': False, 'sdpa': 'math', 'batch': args.batch, 'cfg': args.cfg,
                       'warmups': 5, 'trials_seconds': samples, 'device_event_seconds': device_samples, 'cleanup_seconds': cleanup_samples,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'outputs': signature})
    print(json.dumps({'batch': args.batch, 'cfg': args.cfg, 'output_sha256': digest_file(output)}, sort_keys=True))


if __name__ == '__main__':
    main()
