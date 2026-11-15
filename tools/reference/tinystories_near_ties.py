"""Summarize near ties in TinyStories teacher traces without changing an oracle."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--comparison-dir', type=Path, help='directory holding comparison summaries')
    parser.add_argument('--comparison-prefix', default='rust')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must not exist')
    result = {}
    for seed in (100, 101, 102, 103, 104):
        comparison_dir = args.comparison_dir or args.run
        stats = json.loads((comparison_dir / f'{args.comparison_prefix}-{seed}.json').read_text())
        rows = {}
        with safe_open(str(args.run / f'python-{seed}' / 'trace.safetensors'), framework='np') as trace:
            names = [name for name in trace.keys()
                     if (name.startswith('generator.level_') and name.endswith('.logits'))
                     or name == 'generator.final_logits']
            if len(names) != 12:
                raise ValueError('expected eleven levels and final logits')
            for name in names:
                logits = trace.get_tensor(name)
                values = logits.reshape(-1, logits.shape[-1])
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite reference logits')
                top = np.partition(values, -2, axis=1)[:, -2:]
                margin = top[:, 1] - top[:, 0]
                bound = 2 * stats[name]['max_abs']
                rows[name] = {'rows': len(margin), 'minimum_margin': float(margin.min()),
                              'within_twice_tensor_max_abs_error': int((margin <= bound).sum()),
                              'bound': bound}
        result[str(seed)] = rows
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
