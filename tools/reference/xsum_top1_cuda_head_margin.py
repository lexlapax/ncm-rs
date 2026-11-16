"""Inspect the first divergent head choice on the two-row CUDA teacher trace."""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
from xsum_artifact import digest_file

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / 'reports/phase4-xsum-top1-divergence-cuda-v1/trace.safetensors'
METAL = ROOT / 'reports/phase4-xsum-mlx-full-top1-captured-v1'
TRACE_SHA = '42dc76b4ed384a76f3dc375890d86c0e33f8ad74a1e2b0753ba3c3f6524d1895'
METAL_MANIFEST_SHA = 'b0e6efa956268e3e424ac5a37a11f60b860bcedf8b2c1ac7661a2f4dc4d91923'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if digest_file(TRACE) != TRACE_SHA or digest_file(METAL / 'manifest.json') != METAL_MANIFEST_SHA:
        raise ValueError('trace or Metal manifest differs')
    trace = load_file(TRACE)
    row, global_pos = 4937, 8
    envelope = json.loads((METAL / f'batch-{row // 32 * 32:05}.json').read_bytes())
    import hashlib
    raw = json.dumps(envelope['record'], sort_keys=True, separators=(',', ':')).encode()
    if envelope['sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Metal receipt differs')
    metal_code = envelope['record']['result']['codes'][row % 32][global_pos]
    python_code = int(trace['generator.forced_codes'][1, global_pos])
    values = trace['generator.level_01.logits'][1, 4]
    if tuple(values.shape) != (4096,) or (metal_code, python_code) != (3690, 3182):
        raise ValueError('unexpected candidate identities')
    result = {'row': row, 'global_position': global_pos, 'level': 1, 'level_position': 4,
              'python_saved_code': python_code, 'metal_code': metal_code,
              'cuda_teacher_argmax': int(values.argmax()),
              'cuda_teacher_python_logit': float(values[python_code]),
              'cuda_teacher_metal_logit': float(values[metal_code]),
              'cuda_teacher_metal_minus_python': float(values[metal_code] - values[python_code]),
              'python_rank_strict': int((values > values[python_code]).sum()) + 1,
              'scope': 'batch-2 CUDA teacher with forced Python history; saved outputs used batch 32'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True); out.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
