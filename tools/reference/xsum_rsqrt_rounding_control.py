"""Check correctly rounded reciprocal square roots against a captured CUDA trace."""
import hashlib
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

path = Path('reports/phase4-xsum-block-replay-v1/candidate-norm-variance-input.safetensors')
expected_hash = 'c6c4c015f59f2bb4c6fbc680a3d66e6f07a05282643ea8aa6d7c27b2e8b0cb54'
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
    raise ValueError('captured input digest differs')
d = load_file(path)
variance, cuda = d['variance'].reshape(-1), d['rstd'].reshape(-1)
if variance.shape != (518,) or cuda.shape != (518,) or not np.isfinite(variance).all() or not np.isfinite(cuda).all():
    raise ValueError('unexpected captured input')
inputs = np.add(variance, np.float32(1e-6), dtype=np.float32)
rounded = (1.0 / np.sqrt(inputs.astype(np.float64))).astype(np.float32)
with localcontext() as ctx:
    ctx.prec = 80
    for x, y in zip(inputs, rounded, strict=True):
        value = Decimal(1) / Decimal.from_float(float(x)).sqrt()
        center = Decimal.from_float(float(y))
        lower = Decimal.from_float(float(np.nextafter(y, np.float32(-np.inf))))
        upper = Decimal.from_float(float(np.nextafter(y, np.float32(np.inf))))
        if not ((lower + center) / 2 < value < (center + upper) / 2):
            raise ValueError('FP64 result not demonstrably correctly rounded')
diff = rounded.view(np.int32) - cuda.view(np.int32)
values, counts = np.unique(diff, return_counts=True)
print('inputs', len(inputs), 'ulp_delta_counts', dict(zip(values.tolist(), counts.tolist(), strict=True)),
      'max_abs', float(np.max(np.abs(rounded - cuda))))
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
    raise ValueError('captured input changed')
