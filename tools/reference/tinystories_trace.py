"""Capture controlled TinyStories teacher-forced traces from pinned Python CUDA."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinystories_artifact import INPUTS, UPSTREAM, verify_input, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample', type=Path, help='development samples supplying fixed code rows')
    parser.add_argument('--count', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output must not exist')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fallback')
    rev = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()
    if rev != UPSTREAM or dirty:
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, sha) in INPUTS.items():
        verify_input(args.source / name, size, sha)
    cfg = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    if cfg.task != 'tinystories':
        raise ValueError('incorrect task')
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
    # Upstream eval constructs codebook-tied heads after moving the codec to CUDA.
    codec = codec.cuda().eval()
    model = NextConceptTransformer(codec, cfg.model).cuda()
    model.load_state_dict(generator_state['model'], strict=True)
    model.eval()
    traces = {}
    def save(name, value):
        traces[name] = value.detach().cpu().contiguous().clone()
    levels = list(codec.vq.levels)
    if args.sample is None:
        batch = 2
        forced = [((torch.arange(batch * length, device='cuda').reshape(batch, length) * 37 + level * 97) % 4096).long()
                  for level, length in enumerate(levels)]
    else:
        rows = json.loads(args.sample.read_bytes())['codes'][:args.count]
        if not 1 <= len(rows) <= 16 or any(len(row) != sum(levels) or any(type(v) is not int or not 0 <= v < 4096 for v in row) for row in rows):
            raise ValueError('invalid fixed development codes')
        batch = len(rows)
        combined = torch.tensor(rows, dtype=torch.long, device='cuda')
        forced = list(combined.split(levels, dim=1))
    prefix = torch.zeros((batch, 1), dtype=torch.long, device='cuda')
    # Synthetic codec IDs cover ordinary IDs and special tokens; not natural text quality.
    inputs = (torch.arange(batch * 128, device='cuda').reshape(batch, 128) * 29 % 8192).long()
    calls = [0]
    def forced_sample(logits, temperature, top_k, top_p):
        level = calls[0] + 1
        save(f'generator.level_{level:02d}.logits', logits)
        assert tuple(logits.shape) == (batch, levels[level], 4096)
        calls[0] += 1
        return forced[level]
    model._sample_level_raw = forced_sample
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        save('codec.input_ids', inputs)
        save('codec.pre_quant', codec.encode_z(inputs))
        codes = codec.encode(inputs)
        save('codec.codes', codes)
        save('codec.logits', codec.decode(codes))
        save('generator.prefix_ids', prefix)
        generated = model.generate(forced[0], prefix, temperature=1.0, top_k=0, top_p=1.0,
                                   use_kv_cache=False, cfg_scale=0.0)
        assert calls[0] == 11 and torch.equal(generated, torch.cat(forced, dim=1))
        save('generator.forced_codes', generated)
        z_hat = torch.zeros(batch, levels[-1], codec.embed_dim, device='cuda')
        for level, codes in enumerate(forced):
            z_hat, _ = codec.vq.get_next_autoregressive_input(level, z_hat, codec.vq.embeddings[level](codes))
            save(f'reconstruction.level_{level:02d}', z_hat)
        final = codec.decoder(z_hat)
        save('generator.final_logits', final)
        save('generator.output_ids', final.argmax(-1))
        torch.cuda.synchronize()
    # Create a new directory only after successful capture; it contains project data only.
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(traces, str(args.output / 'trace.safetensors'))
    report = {'upstream': rev, 'sources': {k: v[1] for k, v in INPUTS.items()},
              'torch': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
              'tf32': False, 'sdpa': 'math', 'batch': batch,
              'mode': 'teacher forced fixed code indices; sampling not evaluated',
              'sample_sha256': digest_file(args.sample) if args.sample else None,
              'trace_sha256': digest_file(args.output / 'trace.safetensors'),
              'tensors': sorted(traces)}
    with (args.output / 'report.json').open('x') as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write('\n')


if __name__ == '__main__':
    main()
