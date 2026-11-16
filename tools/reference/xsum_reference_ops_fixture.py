"""Capture the explicit XSum CUDA operator fixture without model weights."""
import argparse
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as functional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if torch.__version__ != '2.15.0+cu132' or not torch.cuda.is_available():
        raise ValueError('the pinned PyTorch CUDA environment is required')
    if torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3060 Ti':
        raise ValueError('the recorded fixture GPU is required')
    torch.backends.cuda.matmul.allow_tf32 = False

    def values(n):
        return torch.tensor([((i % 97) - 48) / 32 for i in range(n)], device='cuda', dtype=torch.float32)

    def weights(n):
        return torch.tensor([1 + ((i % 11) - 5) / 128 for i in range(n)], device='cuda', dtype=torch.float32)

    def bias(n):
        return torch.tensor([((i % 7) - 3) / 64 for i in range(n)], device='cuda', dtype=torch.float32)

    def bits(tensor):
        raw = tensor.detach().contiguous().cpu().numpy().tobytes()
        return list(struct.unpack('<' + 'I' * tensor.numel(), raw))

    with torch.inference_mode():
        x = values(768).reshape(1, 1, 768)
        layernorm = functional.layer_norm(x, [768], weights(768), bias(768), 1e-6)
        gelu = functional.gelu(x)
        q = values(12 * 64).reshape(1, 12, 1, 64)
        rmsnorm = (q * (q.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()) * weights(64)
        cosine = torch.full((1, 32), 0.8, device='cuda')
        sine = torch.full((1, 32), 0.6, device='cuda')
        cosine = torch.cat([cosine, cosine], dim=-1)
        sine = torch.cat([sine, sine], dim=-1)
        rope = q * cosine + torch.cat([-q[..., 32:], q[..., :32]], dim=-1) * sine
        grouped = values(12 * 384).reshape(1, 12, 384)
        groupnorm = functional.group_norm(grouped.transpose(1, 2).contiguous(), 8, eps=1e-5)
        groupnorm = groupnorm.transpose(1, 2).contiguous()
    outputs = [('layernorm', layernorm), ('rmsnorm', rmsnorm), ('groupnorm', groupnorm),
               ('gelu', gelu), ('rope', rope)]
    fixture = {
        'schema_version': 1, 'torch_version': torch.__version__, 'device': 'RTX 3060 Ti',
        'input_rule': '((i mod 97)-48)/32', 'weight_rule': '1+((i mod 11)-5)/128',
        'bias_rule': '((i mod 7)-3)/64', 'bits': {name: bits(value) for name, value in outputs},
    }
    with args.output.open('x') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
