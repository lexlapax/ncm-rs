"""Capture finite tied XSum heads from the pinned CUDA sampling sequence."""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError('new output and CUDA required')
    if (torch.__version__ != '2.15.0+cu132'
            or torch.version.git_version != 'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e'):
        raise ValueError('unexpected pinned PyTorch build')
    generator = random.Random(784162)
    counts = (2, 3, 4, 8, 16, 64, 512, 4096)
    ties = [sorted(generator.sample(range(4096), count)) for count in counts for _ in range(128)]
    logits = torch.full((len(ties), 4096), -10.0, device='cuda')
    for row, indices in enumerate(ties):
        logits[row, indices] = -0.0 if row % 2 else 0.0
        if len(indices) > 1 and row % 3 == 0:
            logits[row, indices[-1]] = 0.0
    torch.manual_seed(42)
    kth = torch.topk(logits, 1, dim=-1).values[:, -1, None]
    filtered = logits.masked_fill(logits < kth, -torch.inf)
    values, indices = torch.sort(filtered, dim=-1, descending=True)
    probs = F.softmax(values, dim=-1)
    remove = torch.cumsum(probs, dim=-1) - probs > 0.0
    values = values.masked_fill(remove, -torch.inf)
    restored = torch.zeros_like(filtered).scatter_(-1, indices, values)
    selected = torch.multinomial(F.softmax(restored, dim=-1), 1).flatten()
    if selected.tolist() != [row[0] for row in ties]:
        raise ValueError('pinned reference does not select the first tied index')
    args.output.mkdir(parents=True)
    save_file({'logits': logits.cpu().contiguous(), 'selected': selected.cpu().contiguous()},
              str(args.output / 'reference.safetensors'))
    with (args.output / 'report.json').open('x') as output:
        json.dump({'torch': torch.__version__, 'torch_git': torch.version.git_version,
                   'device': torch.cuda.get_device_name(0), 'rows': len(ties),
                   'tie_counts': list(counts), 'seed': 42}, output, sort_keys=True)
        output.write('\n')


if __name__ == '__main__':
    main()
