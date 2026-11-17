"""Compare same-batch forced-history CUDA and Metal XSum row traces."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from xsum_artifact import digest_file
from xsum_numpy_selection import select

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final/inference.safetensors'
MANIFEST = ROOT / 'artifacts/canonical/xsum-inference-rope-v2-final/manifest.json'
MANIFEST_SHA = '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'
CUDA_SHA = {4883: 'dcb52412e2d548d858d9ca742adfc4101db6f289760f6647e44131f955633236',
            4937: '69f7e1a07332738f9448eb9682010af06c2223b78971a1a199d978af12ff8fc8'}


def scores_at(logits, unigram, history, position):
    length, vocab = logits.shape
    if position != len(history) or not 0 <= position < length or len(unigram) != vocab:
        raise ValueError('invalid penalty history')
    counts = np.bincount(np.asarray(history, dtype=np.int64), minlength=vocab).astype(np.float64)
    allowance = unigram.astype(np.float64) * length
    scores = logits[position].astype(np.float64)
    scores -= 2.0 * np.maximum(counts - allowance, 0.0)
    scores += ((length - position) / length) * np.maximum(allowance - counts, 0.0)
    for offset, penalty in enumerate((5.0, 2.5, 1.25, 0.625)):
        if offset < position:
            scores[history[-offset - 1]] -= penalty
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', type=int, choices=(4883, 4937), required=True)
    parser.add_argument('--metal-trace', type=Path, required=True)
    parser.add_argument('--metal-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    cuda_trace = ROOT / f'reports/phase4-xsum-top1-row-cuda-v1/row-{args.row}.safetensors'
    cuda_result = ROOT / f'reports/phase4-xsum-top1-row-cuda-v1/row-{args.row}.json'
    if digest_file(cuda_trace) != CUDA_SHA[args.row] or digest_file(MANIFEST) != MANIFEST_SHA:
        raise ValueError('CUDA trace or artifact manifest differs')
    manifest = json.loads(MANIFEST.read_bytes())
    if digest_file(INFERENCE) != manifest['files']['inference.safetensors']['sha256']:
        raise ValueError('inference payload differs')
    watched = [cuda_trace, cuda_result, args.metal_trace, args.metal_result, INFERENCE, MANIFEST, Path(__file__).resolve()]
    before = {str(path): digest_file(path) for path in watched}
    cuda, metal = load_file(cuda_trace), load_file(args.metal_trace)
    expected = {'generator.final_logits'} | {f'generator.level_{i:02}.logits' for i in range(9)}
    if set(cuda) != expected or set(metal) != expected:
        raise ValueError('trace tensor inventory differs')
    report = {}
    for name in sorted(expected):
        a, b = cuda[name].numpy(), metal[name].numpy()
        if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'invalid tensor {name}')
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        entry = {'elements': int(a.size), 'max_abs': float(delta.max()),
                 'outside_existing_band': int((delta > 5e-4 + 2e-4 * np.abs(a.astype(np.float64))).sum())}
        if name.startswith('generator.level_'):
            entry['argmax_differences'] = int((a.argmax(axis=-1) != b.argmax(axis=-1)).sum())
        report[name] = entry
    unigram = load_file(INFERENCE)['token_unigram'].numpy()
    cuda_logits, metal_logits = cuda['generator.final_logits'].numpy(), metal['generator.final_logits'].numpy()
    cuda_ids = select(cuda_logits, unigram)[0].tolist()
    metal_ids = select(metal_logits, unigram)[0].tolist()
    if cuda_ids != json.loads(cuda_result.read_bytes())['selected_ids'] or metal_ids != json.loads(args.metal_result.read_bytes())['selected_ids']:
        raise ValueError('independent FP64 selector differs from native output')
    positions = [i for i, (a, b) in enumerate(zip(cuda_ids, metal_ids)) if a != b]
    first = None
    if positions:
        t = positions[0]
        if cuda_ids[:t] != metal_ids[:t]:
            raise ValueError('first difference has unequal history')
        cs = scores_at(cuda_logits[0], unigram, cuda_ids[:t], t)
        ms = scores_at(metal_logits[0], unigram, cuda_ids[:t], t)
        ci, mi = cuda_ids[t], metal_ids[t]
        first = {'position': t, 'cuda_id': ci, 'metal_id': mi,
                 'cuda_score_difference': float(cs[ci] - cs[mi]),
                 'metal_score_difference': float(ms[ci] - ms[mi])}
    if args.row == 4937:
        a = cuda['generator.level_01.logits'].numpy()[0, 4]
        b = metal['generator.level_01.logits'].numpy()[0, 4]
        report['first_free_generation_code_difference'] = {'level': 1, 'position': 4,
            'cuda_argmax': int(a.argmax()), 'metal_argmax': int(b.argmax()),
            'cuda_3182_minus_3690': float(a[3182] - a[3690]),
            'metal_3182_minus_3690': float(b[3182] - b[3690])}
    if any(digest_file(Path(path)) != sha for path, sha in before.items()):
        raise ValueError('trace inputs changed during comparison')
    result = {'row': args.row, 'input_sha256': before, 'trace': report,
              'id_difference_positions': positions, 'first_id_difference_same_history': first,
              'scope': 'same 32-row batch and forced CUDA code histories; not free Metal generation'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True); out.write('\n')
    print(json.dumps(result, sort_keys=True))
    if any(v.get('outside_existing_band', 0) for v in report.values()):
        raise SystemExit('existing numerical band failed; result retained')


if __name__ == '__main__':
    main()
