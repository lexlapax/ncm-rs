"""Capture pinned XSum first-level cascade and refinement on CUDA."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file

from xsum_artifact import INPUTS, UPSTREAM, verify_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError('output must be new and CUDA must be available')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM
            or subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('upstream checkout must be clean and pinned')
    for name, (size, digest) in INPUTS.items():
        verify_input(args.source / name, size, digest)
    config = OmegaConf.create(yaml.safe_load((args.source / 'config.yaml').read_bytes()))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.vqvae import VQVAE
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(args.source / 'tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    codec = VQVAE(**config.tokenizer.vqvae)
    codec.load_state_dict(state['model'], strict=True)
    codec = codec.cuda().eval()
    tensors = {}

    def save(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()

    with torch.inference_mode():
        ids = torch.tensor([[0, 37, 2048, 4095], [123, 456, 789, 42]], device='cuda')
        up = codec.vq.embeddings[0](ids)
        save('embedding', up)
        up = up.transpose(1, 2)
        for i, layer in enumerate(codec.vq.learned_up_l1):
            up = layer(up)
            save(f'up.{i}', up.transpose(1, 2))
        save('refined', codec.vq._apply_quant_resi(up.transpose(1, 2).contiguous(), 0))
        values = (torch.arange(2 * 48 * 384, device='cuda', dtype=torch.float32).remainder(127) - 63) / 64
        down = values.reshape(2, 48, 384)
        save('down.input', down)
        down = down.transpose(1, 2)
        for i, layer in enumerate(codec.vq.learned_down_l1):
            down = layer(down)
            save(f'down.{i}', down.transpose(1, 2))
        torch.cuda.synchronize()
    for name, (size, digest) in INPUTS.items():
        verify_input(args.source / name, size, digest)
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(args.output / 'trace.safetensors'))
    digest = hashlib.sha256((args.output / 'trace.safetensors').read_bytes()).hexdigest()
    (args.output / 'report.json').write_text(json.dumps({
        'upstream': UPSTREAM, 'tf32': False, 'cuda': torch.cuda.get_device_name(0),
        'trace_sha256': digest, 'tensors': sorted(tensors),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
