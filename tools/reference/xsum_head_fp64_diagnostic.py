import hashlib
import numpy as np
from safetensors.numpy import load_file

paths = {
    'cuda': ('reports/phase4-xsum-mlp-cuda-v1/row-4937.safetensors', '3723d9fda929e4d503614961f2f2126a18b6cefa563accb51c9cb3b8eb54ce0f'),
    'metal': ('reports/phase4-xsum-refined-norm-row-captured-v1/row-4937.safetensors', '9147eaf4bf36060151ae91e53cfe17fdafbccb82c41578fd69916d76a37b863c'),
    'head': ('reports/phase4-xsum-head-chain-input-v1.safetensors', 'eb7ec6d50460e4521ce06e5f5e827ba9fc35e8001a72e8098bd68499a53dc184'),
}
def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
assert all(digest(p) == h for p, h in paths.values())
d = load_file(paths['head'][0])
book = d['book'].astype(np.float64)
weight = d['weight'].astype(np.float64)
scale = np.prod(d['scales'].astype(np.float64))
for name in ('cuda', 'metal'):
    trace = load_file(paths[name][0])
    hidden = trace['diagnostic.level_01.hidden'][0, 4].astype(np.float64)
    captured = trace['generator.level_01.logits'][0, 4]
    print(name, 'captured-f32', 'margin', float(captured[3182] - captured[3690]), 'choice', int(np.argmax(captured)))
    proj = weight @ hidden
    for label, norms in [('recorded-f32', d['norm'].astype(np.float64)), ('recomputed-f64', np.sum(book * book, axis=1))]:
        scores = (2 * (book @ proj) - norms) * scale
        print(name, label, 'margin', format(scores[3182]-scores[3690], '.17g'), 'choice', int(np.argmax(scores)))
assert all(digest(p) == h for p, h in paths.values())
