"""Diagnose the first WMT14 CFG divergence in its original batch."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open

PY_TRACE = 'f3914d2d837325966226cbd4bdfecb7a66242c52971ee8757f3aa9d9df8ea312'
PY_FULL = '78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256'
RUST_FULL = '531fb022c249f097b9c122dfb3a7bdf57e95d898c89d5fd88b40b2cf34e4f88e'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def analyze(a, b, expected_python, expected_rust):
    if a.get_tensor('generator.codes').tolist() != [row['codes'] for row in expected_python] or a.get_tensor('generator.greedy_ids').tolist() != [row['raw_ids'] for row in expected_python]:
        raise ValueError('Python same-batch output differs from full capture')
    if b.get_tensor('generator.codes').tolist() != [row['codes'] for row in expected_rust] or b.get_tensor('generator.greedy_ids').tolist() != [row['raw_ids'] for row in expected_rust]:
        raise ValueError('Rust same-batch output differs from full capture')
    if not torch.equal(a.get_tensor('generator.codes')[:, :47], b.get_tensor('generator.codes')[:, :47]):
        raise ValueError('earlier histories differ')
    levels = []
    for level in range(6):
        x = a.get_tensor(f'generator.level_{level:02}.logits')
        y = b.get_tensor(f'generator.level_{level:02}.logits')
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('head shape or finiteness differs')
        errors = (x.double() - y.double()).abs()
        levels.append({'level': level, 'elements': x.numel(), 'max_abs_error': errors.max().item(),
                       'outside_accumulated_band': (errors > 0.0005 + 0.0002 * x.double().abs()).sum().item(),
                       'argmax_differences': (x.argmax(-1) != y.argmax(-1)).sum().item()})
    x, y = a.get_tensor('generator.level_05.logits')[1, 6], b.get_tensor('generator.level_05.logits')[1, 6]
    return {'levels': levels, 'row': 2181, 'level': 5, 'position': 6,
            'python_choice': x.argmax().item(), 'rust_choice': y.argmax().item(),
            'python_scores': {str(i): x[i].item() for i in (32, 1071)},
            'rust_scores': {str(i): y[i].item() for i in (32, 1071)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('python_trace', type=Path)
    p.add_argument('rust_trace', type=Path)
    p.add_argument('python_full', type=Path)
    p.add_argument('rust_full', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--rust-trace-sha256', required=True)
    args = p.parse_args()
    watched = [(args.python_trace, PY_TRACE), (args.rust_trace, args.rust_trace_sha256),
               (args.python_full, PY_FULL), (args.rust_full, RUST_FULL)]
    if args.output.exists() or args.output.is_symlink() or any(digest(path) != sha for path, sha in watched):
        raise ValueError('output exists or input hash differs')
    py = json.loads(args.python_full.read_bytes())['outputs'][2180:2184]
    rs = json.loads(args.rust_full.read_bytes())['outputs'][2180:2184]
    with safe_open(args.python_trace, framework='pt', device='cpu') as a, safe_open(args.rust_trace, framework='pt', device='cpu') as b:
        result = analyze(a, b, py, rs)
    if any(digest(path) != sha for path, sha in watched):
        raise ValueError('input changed during comparison')
    result.update({'scope': 'same-batch free histories until first CFG divergence; not a gate waiver',
                   'python_trace_sha256': PY_TRACE, 'rust_trace_sha256': args.rust_trace_sha256})
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
