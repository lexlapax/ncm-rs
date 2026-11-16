"""Compare saved same-logit sampler choices with an independent FP64 inverse CDF."""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
LOGITS = ROOT / 'reports/phase4-xsum-shared-history-cuda-v1/logits.safetensors'
CUDA = ROOT / 'reports/phase4-xsum-mlx-stochastic-cuda-probe-v1.json'
METAL = ROOT / 'reports/phase4-xsum-shared-history-captured-v1/result.json'
FIXED = ROOT / 'reports/phase4-xsum-mlx-fixedpoint-captured-v1/result.json'
EXPECTED = {
    LOGITS: 'c57f4f6c9a88e0f4d247dc7bc6fa6cb525e3c87945205fa33b0535af2de74018',
    CUDA: 'f9e9aa42b6bcd4b8fd405e50c9689b11f0f3871bf1ec69c0535ef29b2d2c7f08',
    METAL: '62d6997d5737c182ddbe9a2d47f8c0a5f2b9b9b011d354df7650a1129b60b30d',
    FIXED: 'd93238f12ede0ac2a9535c6e9bb8c651f9bc3cf829f49f863ae0aeeec146e2ec',
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if any(digest(path) != sha for path, sha in EXPECTED.items()):
        raise ValueError('input hash mismatch')
    cuda = json.loads(CUDA.read_bytes())['codes']
    metal = json.loads(METAL.read_bytes())['differences']
    fixed = json.loads(FIXED.read_bytes())['differences']
    old_map = {(x['row'], x['flat_position']): x['metal_same_logits'] for x in metal}
    fixed_map = {(x['row'], x['flat_position']): x['actual'] for x in fixed}
    errors = {'cuda': [], 'metal_old': [], 'fixedpoint': []}
    mask = (1 << 64) - 1
    state = 42
    offset = 0
    with safe_open(LOGITS, framework='numpy') as tensors:
        for level in range(9):
            values = tensors.get_tensor(f'generator.level_{level:02}.logits').astype(np.float64)
            if values.ndim != 3 or values.shape[0] != 32 or values.shape[2] != 4096 or not np.isfinite(values).all():
                raise ValueError('invalid logits')
            length = values.shape[1]
            words = []
            for _ in range(32 * length):
                state = (state + 0x9e3779b97f4a7c15) & mask
                z = ((state ^ (state >> 30)) * 0xbf58476d1ce4e5b9) & mask
                z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
                words.append((z ^ (z >> 31)) >> 32)
            flat = values.reshape(-1, 4096)
            weights = np.exp(flat - flat.max(axis=1, keepdims=True))
            cumulative = np.cumsum(weights, axis=1)
            target = np.asarray(words, dtype=np.float64) / 2**32 * cumulative[:, -1]
            choices = np.sum(cumulative <= target[:, None], axis=1)
            for row in range(32):
                for pos in range(length):
                    index = offset + pos
                    expected = int(choices[row * length + pos])
                    baseline = cuda[row * 200 + index]
                    for name, actual in (('cuda', baseline), ('metal_old', old_map.get((row, index), baseline)),
                                         ('fixedpoint', fixed_map.get((row, index), baseline))):
                        if actual != expected:
                            errors[name].append({'row': row, 'position': index, 'actual': actual, 'fp64': expected})
            offset += length
    if offset != 200 or any(digest(path) != sha for path, sha in EXPECTED.items()):
        raise ValueError('inputs changed or code lengths differ')
    print(json.dumps({'schema_version': 1, 'scope': 'independent FP64 inverse CDF on saved CUDA logits and same words',
                      'draws': 6400, 'input_sha256': {str(k.relative_to(ROOT)): v for k, v in EXPECTED.items()},
                      'differences': errors}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
